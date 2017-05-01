
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>

using namespace std;
using namespace Teuchos;


int main(int argc, char *argv[]) {

#ifdef HAVE_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif
  cout << Comm <<endl;

  long initsize  = pow(50,3);
  long finalsize = pow(400,3);
  int base = 50;

  std::ofstream fout;
  struct timeval tim;
  fout.open("MLAztecOOEpetra.out");

  for(long i = initsize; i<=finalsize;){
	for(int j = 0; j <4; j++){
		SolverControl solver_control(5000, 1e-6 * src.block(1).l2_norm());

	  cout << "Solver performed " << solver.NumIters() << " iterations." << endl
		   << "Norm of true residual = " << solver.TrueResidual() << endl;
      fout << t2-t1 <<","<< solver.NumIters() << ","<< solver.TrueResidual()  << std::endl;
	}
    base += 50;
	i = pow(base,3);
  }
  #ifdef HAVE_MPI
   MPI_Finalize() ;
  #endif
  return 0;
}s
