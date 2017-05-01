// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2014 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
// 
// @Modified By Shilei Lin 2017/04/30
// ---------------------------------- -----------------------------------

#include <deal.II/lac/trilinos_solver.h>

#ifdef DEAL_II_WITH_TRILINOS

#  include <deal.II/base/conditional_ostream.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#  include <deal.II/lac/trilinos_vector_base.h>
#  include <deal.II/lac/trilinos_precondition.h>

#  include <cmath>

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers
{

  SolverBase::AdditionalData::AdditionalData (const bool         output_solver_details,
                                              const unsigned int gmres_restart_parameter)
    :
    output_solver_details (output_solver_details),
    gmres_restart_parameter (gmres_restart_parameter)
  {}
  SolverBase::SolverBase (SolverControl  &cn)
    :
    solver_name    (gmres),
    solver_control (cn)
  {}
  SolverBase::SolverBase (const enum SolverBase::SolverName  solver_name,
                          SolverControl                     &cn)
    :
    solver_name    (solver_name),
    solver_control (cn)
  {}
  SolverBase::~SolverBase ()
  {}
  SolverControl &
  SolverBase::control() const
  {
    return solver_control;
  }

  double
  SolverBase::solve (const SparseMatrix     &a,
                       VectorBase             &x,
                       const VectorBase       &b,
                       const PreconditionBase &preconditioner)
    {
      linear_problem.reset();

      // We need an Belos::LinearProblem<double,MV,OP> object to let the Belos solver know
      // about the matrix and vectors.
      RCP<const Epetra_CrsMatrix> A = rcpFromRef(*(const_cast<Epetra_CrsMatrix *>(&a.trilinos_matrix())));
	  RCP<MV> X = rcpFromRef(*(const_cast<Epetra_FEVector *>(&x.trilinos_vector())));
	  RCP<MV> B = rcp (new MV (A->OperatorDomainMap(), 1));	  

	  A->Apply(*X,*B);
	       
      B = rcpFromRef(*(const_cast<MV *>(&b.trilinos_vector())));
      linear_problem = rcp(new Belos::LinearProblem<double,MV,OP>(A, X, B));

      return do_solve(preconditioner);
    }

  double
  SolverBase::solve (Epetra_Operator        &a,
                     VectorBase             &x,
                     const VectorBase       &b,
                     const PreconditionBase &preconditioner)
  {
	linear_problem.reset();

    // We need an Belos::LinearProblem<double,MV,OP> object to let the Belos solver know
    // about the matrix and vectors.
    RCP<const Epetra_Operator> A = rcpFromRef(*(const_cast<Epetra_Operator *>(&a)));
    RCP<MV> X = rcpFromRef(*(const_cast<Epetra_FEVector *>(&x.trilinos_vector())));
	RCP<MV> B = rcp (new MV (A->OperatorDomainMap(), 1));	  
		  
	A->Apply(*X,*B);

    B = rcpFromRef(*(const_cast<MV *>(&b.trilinos_vector())));

    //Set Properties
    linear_problem = rcp(new Belos::LinearProblem<double,MV,OP>(A, X, B));
    return do_solve(preconditioner);
  }

  double
  SolverBase::solve (const SparseMatrix           &A,
                     dealii::Vector<double>       &x,
                     const dealii::Vector<double> &b,
                     const PreconditionBase       &preconditioner)
  {
    linear_problem.reset();

    // In case we call the solver with deal.II vectors, we create views of the
    // vectors in Epetra format.
    Assert (x.size() == A.n(),
            ExcDimensionMismatch(x.size(), A.n()));
    Assert (b.size() == A.m(),
            ExcDimensionMismatch(b.size(), A.m()));
    Assert (A.local_range ().second == A.m(),
            ExcMessage ("Can only work in serial when using deal.II vectors."));
    Assert (A.trilinos_matrix().Filled(),
            ExcMessage ("Matrix is not compressed. Call compress() method."));
            
    Epetra_Vector ep_x (View, A.domain_partitioner(), x.begin());
    Epetra_Vector ep_b (View, A.range_partitioner(), const_cast<double *>(b.begin()));
    
	RCP<const Epetra_CrsMatrix> eA = rcpFromRef(*(const_cast<Epetra_CrsMatrix *>(&A.trilinos_matrix())));
	RCP<MV> X = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_x)));
	RCP<MV> B = rcp (new MV(eA->OperatorDomainMap(), 1));
	  
	eA->Apply(*X,*B);
	   
    B = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_b)));

    // We need an Belos::LinearProblem<double,MV,OP> object to let the Belos solver know
    // about the matrix and vectors.
    linear_problem = rcp(new Belos::LinearProblem<double,MV,OP>(eA, X, B));
    return do_solve(preconditioner);
  }

  double
  SolverBase::solve (Epetra_Operator              &A,
                     dealii::Vector<double>       &x,
                     const dealii::Vector<double> &b,
                     const PreconditionBase       &preconditioner)
  {
    linear_problem.reset();

    Epetra_Vector ep_x (View, A.OperatorDomainMap(), x.begin());
    Epetra_Vector ep_b (View, A.OperatorRangeMap(), const_cast<double *>(b.begin()));

    // We need an Belos::LinearProblem<double,MV,OP> object to let the AztecOO solver know
    // about the matrix and vectors.

	RCP<const Epetra_Operator> eA = rcpFromRef(*(const_cast<Epetra_Operator *>(&A)));
	RCP<MV> X = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_x)));
	RCP<MV> B = rcp (new MV(eA->OperatorDomainMap (), 1));
	  
	eA->Apply(*X,*B);
	   
    B = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_b)));

	// We need an Belos::LinearProblem<double,MV,OP> object to let the Belos solver know
	// about the matrix and vectors.
	linear_problem = rcp(new Belos::LinearProblem<double,MV,OP>(eA, X, B));
    return do_solve(preconditioner);
  }

  double
  SolverBase::solve (const SparseMatrix                                  &A,
                     dealii::parallel::distributed::Vector<double>       &x,
                     const dealii::parallel::distributed::Vector<double> &b,
                     const PreconditionBase                              &preconditioner)
  {
    linear_problem.reset();

    // In case we call the solver with deal.II vectors, we create views of the
    // vectors in Epetra format.
    AssertDimension (static_cast<TrilinosWrappers::types::int_type>(x.local_size()),
                     A.domain_partitioner().NumMyElements());
    AssertDimension (static_cast<TrilinosWrappers::types::int_type>(b.local_size()),
                     A.range_partitioner().NumMyElements());

    Epetra_Vector ep_x (View, A.domain_partitioner(), x.begin());
    Epetra_Vector ep_b (View, A.range_partitioner(), const_cast<double *>(b.begin()));

	RCP<const Epetra_CrsMatrix> eA = rcpFromRef(*(const_cast<Epetra_CrsMatrix *>(&A.trilinos_matrix())));
	RCP<MV> X = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_x)));
	RCP<MV> B = rcp (new MV(eA->OperatorDomainMap (), 1));
	  
	eA->Apply(*X,*B);
	   
    B = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_b)));
	// We need an Belos::LinearProblem<double,MV,OP> object to let the Belos solver know
	// about the matrix and vectors.
	//Set Properties
	linear_problem = rcp(new Belos::LinearProblem<double,MV,OP>(eA, X, B));
    return do_solve(preconditioner);
  }

  double
  SolverBase::solve (Epetra_Operator                                     &A,
                     dealii::parallel::distributed::Vector<double>       &x,
                     const dealii::parallel::distributed::Vector<double> &b,
                     const PreconditionBase                              &preconditioner)
  {
    linear_problem.reset();

    AssertDimension (static_cast<TrilinosWrappers::types::int_type>(x.local_size()),
                     A.OperatorDomainMap().NumMyElements());
    AssertDimension (static_cast<TrilinosWrappers::types::int_type>(b.local_size()),
                     A.OperatorRangeMap().NumMyElements());

    Epetra_Vector ep_x (View, A.OperatorDomainMap(), x.begin());
    Epetra_Vector ep_b (View, A.OperatorRangeMap(), const_cast<double *>(b.begin()));
	
	RCP<const Epetra_Operator> eA = rcpFromRef(*(const_cast<Epetra_Operator *>(&A)));
	RCP<MV> X = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_x)));
	RCP<MV> B = rcp (new MV(eA->OperatorDomainMap (), 1),false);
	  
	eA->Apply(*X,*B);
	   
    B = rcpFromRef(*(const_cast<Epetra_Vector *>(&ep_b)));
	// We need an Belos::LinearProblem<double,MV,OP> object to let the Belos solver know
	// about the matrix and vectors.
	//Set Properties
	linear_problem = rcp(new Belos::LinearProblem<double,MV,OP>(eA, X, B));
    return do_solve(preconditioner);
  }

  double
  SolverBase::do_solve(const PreconditionBase &preconditioner){
    factory.reset();
    newSolver.reset();
    
	factory = rcp(new Belos::SolverFactory< ST, MV, OP >);
	// ... set some options, ...
	RCP<ParameterList> solverParams = rcp(new ParameterList());
	int max_iters = solver_control.max_steps();
	double tol = solver_control.tolerance();
	
	solverParams->set("Maximum Iterations",    max_iters);
	solverParams->set("Convergence Tolerance", tol);
	//solverParams->set("Verbosity",Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
	//solverParams->set("Output Frequency",      1);
	//solverParams->set("Output Style",          Belos::Brief);

		  
   switch (solver_name){
      case cg:
        newSolver = factory->create("CG", solverParams);
        break;
      case cgs:
        newSolver = factory->create("Block CG", solverParams);
        break;
      case gmres:
        solverParams->set("Maximum Restarts", additional_data.gmres_restart_parameter);
        newSolver = factory->create("GMERS", solverParams);
        //solver.SetAztecOption(AZ_kspace, additional_data.gmres_restart_parameter);
        break;
      case bicgstab:
        newSolver = factory->create("bicgstab", solverParams);
        break;
      case tfqmr:
        newSolver = factory->create("TFQMR", solverParams);
        break;
      default:
        Assert (false, ExcNotImplemented());
     }
     // Introduce the preconditioner, if the identity preconditioner is used,
	 // the precondioner is set to none, ...
	 if (preconditioner.preconditioner.use_count()!=0){
		RCP<Epetra_Operator>MLPrec = //rcp(preconditioner.preconditioner.get());
			rcpFromRef(*(const_cast<Epetra_Operator *>(preconditioner.preconditioner.get())));
	 	RCP<Belos::EpetraPrecOp> RP = rcp(new Belos::EpetraPrecOp(MLPrec));
		linear_problem->setRightPrec(RP);
	 }
	 linear_problem->setProblem();
	 
	 // Next we can allocate the Belos solver...
	 newSolver->setProblem(linear_problem);
	 // ... and then solve!
	 Belos::ReturnType result = newSolver->solve();

	 // report errors in more detail than just by checking whether the return
	 // status is zero or greater. the error strings are taken from the
	 // implementation of the AztecOO::Iterate function
	 if(result == Belos::Unconverged)
		 AssertThrow (false, ExcMessage("Belos::ReturnType Unconverged!"));

	 // Finally, let the deal.II SolverControl object know what has
	 // happened. If the solve succeeded, the status of the solver control will
	 // turn into SolverControl::success. 
	 double actTol = newSolver->achievedTol();
	 solver_control.check (newSolver->getNumIters(), actTol);
     
	 if (solver_control.last_check() != SolverControl::success)
	   AssertThrow(false, SolverControl::NoConvergence (solver_control.last_step(),
														solver_control.last_value()));
														
	 return actTol;
  }



  /* ---------------------- SolverCG ------------------------ */

  SolverCG::AdditionalData::
  AdditionalData (const bool output_solver_details)
    :
    output_solver_details (output_solver_details)
  {}



  SolverCG::SolverCG (SolverControl        &cn,
                      const AdditionalData &data)
    :
    SolverBase (cn),
    additional_data (data.output_solver_details)
  {
    solver_name = cg;
  }


  /* ---------------------- SolverGMRES ------------------------ */

  SolverGMRES::AdditionalData::
  AdditionalData (const bool output_solver_details,
                  const unsigned int restart_parameter)
    :
    output_solver_details (output_solver_details),
    restart_parameter (restart_parameter)
  {}



  SolverGMRES::SolverGMRES (SolverControl        &cn,
                            const AdditionalData &data)
    :
    SolverBase (cn),
    additional_data (data.output_solver_details,
                     data.restart_parameter)
  {
    solver_name = gmres;
  }


  /* ---------------------- SolverBicgstab ------------------------ */

  SolverBicgstab::AdditionalData::
  AdditionalData (const bool output_solver_details)
    :
    output_solver_details (output_solver_details)
  {}




  SolverBicgstab::SolverBicgstab (SolverControl        &cn,
                                  const AdditionalData &data)
    :
    SolverBase (cn),
    additional_data (data.output_solver_details)
  {
    solver_name = bicgstab;
  }


  /* ---------------------- SolverCGS ------------------------ */

  SolverCGS::AdditionalData::
  AdditionalData (const bool output_solver_details)
    :
    output_solver_details (output_solver_details)
  {}




  SolverCGS::SolverCGS (SolverControl        &cn,
                        const AdditionalData &data)
    :
    SolverBase (cn),
    additional_data (data.output_solver_details)
  {
    solver_name = cgs;
  }


  /* ---------------------- SolverTFQMR ------------------------ */

  SolverTFQMR::AdditionalData::
  AdditionalData (const bool output_solver_details)
    :
    output_solver_details (output_solver_details)
  {}



  SolverTFQMR::SolverTFQMR (SolverControl        &cn,
                            const AdditionalData &data)
    :
    SolverBase (cn),
    additional_data (data.output_solver_details)
  {
    solver_name = tfqmr;
  }



  /* ---------------------- SolverDirect ------------------------ */

  SolverDirect::AdditionalData::
  AdditionalData (const bool output_solver_details,
                  const std::string &solver_type)
    :
    output_solver_details (output_solver_details),
    solver_type(solver_type)
  {}




  SolverDirect::SolverDirect (SolverControl  &cn,
                              const AdditionalData &data)
    :
    solver_control (cn),
    additional_data (data.output_solver_details,data.solver_type)
  {}



  SolverDirect::~SolverDirect ()
  {}



  SolverControl &
  SolverDirect::control() const
  {
    return solver_control;
  }



  void
  SolverDirect::do_solve()
  {
    // Fetch return value of Amesos Solver functions
    int ierr;

    // First set whether we want to print the solver information to screen or
    // not.
    ConditionalOStream  verbose_cout (std::cout,
                                      additional_data.output_solver_details);

    solver.reset();

    // Next allocate the Amesos solver, this is done in two steps, first we
    // create a solver Factory and and generate with that the concrete Amesos
    // solver, if possible.
    Amesos Factory;

    AssertThrow(
      Factory.Query(additional_data.solver_type.c_str()),
      ExcMessage (std::string ("You tried to select the solver type <") +
                  additional_data.solver_type +
                  "> but this solver is not supported by Trilinos either "
                  "because it does not exist, or because Trilinos was not "
                  "configured for its use.")
    );

    solver.reset (
      Factory.Create(additional_data.solver_type.c_str(), *linear_problem)
    );

    verbose_cout << "Starting symbolic factorization" << std::endl;
    ierr = solver->SymbolicFactorization();
    AssertThrow (ierr == 0, ExcTrilinosError(ierr));

    verbose_cout << "Starting numeric factorization" << std::endl;
    ierr = solver->NumericFactorization();
    AssertThrow (ierr == 0, ExcTrilinosError(ierr));

    verbose_cout << "Starting solve" << std::endl;
    ierr = solver->Solve();
    AssertThrow (ierr == 0, ExcTrilinosError(ierr));

    // Finally, let the deal.II SolverControl object know what has
    // happened. If the solve succeeded, the status of the solver control will
    // turn into SolverControl::success.
    solver_control.check (0, 0);

    if (solver_control.last_check() != SolverControl::success)
      AssertThrow(false, SolverControl::NoConvergence (solver_control.last_step(),
                                                       solver_control.last_value()));
  }


  void
  SolverDirect::solve (const SparseMatrix     &A,
                       VectorBase             &x,
                       const VectorBase       &b)
  {
    // We need an Belos::LinearProblem<double,MV,OP> object to let the Amesos solver know
    // about the matrix and vectors.
	linear_problem.reset
    (new Epetra_LinearProblem(const_cast<Epetra_CrsMatrix *>(&A.trilinos_matrix()),
                              &x.trilinos_vector(),
                              const_cast<MV*>(&b.trilinos_vector())));

    do_solve();
  }



  void
  SolverDirect::solve (const SparseMatrix           &A,
                       dealii::Vector<double>       &x,
                       const dealii::Vector<double> &b)
  {

    // In case we call the solver with deal.II vectors, we create views of the
    // vectors in Epetra format.
    Assert (x.size() == A.n(),
            ExcDimensionMismatch(x.size(), A.n()));
    Assert (b.size() == A.m(),
            ExcDimensionMismatch(b.size(), A.m()));
    Assert (A.local_range ().second == A.m(),
            ExcMessage ("Can only work in serial when using deal.II vectors."));
    Epetra_Vector ep_x (View, A.domain_partitioner(), x.begin());
    Epetra_Vector ep_b (View, A.range_partitioner(), const_cast<double *>(b.begin()));

    // We need an Belos::LinearProblem<double,MV,OP> object to let the Amesos solver know
    // about the matrix and vectors.
    linear_problem.reset (new Epetra_LinearProblem
                          (const_cast<Epetra_CrsMatrix *>(&A.trilinos_matrix()),
                           &ep_x, &ep_b));

    do_solve();
  }



  void
  SolverDirect::solve (const SparseMatrix                                  &A,
                       dealii::parallel::distributed::Vector<double>       &x,
                       const dealii::parallel::distributed::Vector<double> &b)
  {
    AssertDimension (static_cast<TrilinosWrappers::types::int_type>(x.local_size()),
                     A.domain_partitioner().NumMyElements());
    AssertDimension (static_cast<TrilinosWrappers::types::int_type>(b.local_size()),
                     A.range_partitioner().NumMyElements());
    Epetra_Vector ep_x (View, A.domain_partitioner(), x.begin());
    Epetra_Vector ep_b (View, A.range_partitioner(), const_cast<double *>(b.begin()));

    // We need an Belos::LinearProblem<double,MV,OP> object to let the Amesos solver know
    // about the matrix and vectors.
    linear_problem.reset (new Epetra_LinearProblem
                          (const_cast<Epetra_CrsMatrix *>(&A.trilinos_matrix()),
                           &ep_x, &ep_b));

    do_solve();
  }

}

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PETSC
