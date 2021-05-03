// # include <mpi.h> <iostream> <string> using namespace std;

int main(int argc, char* argv[]) {
  int p, id;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  if (myid == 0) {
    const int bufLen = 100; char greeting[bufLen];
    cout << "Process " << id << " receives greetings from " << p - 1 << " processes!" << endl;
    for (int i = 1; i < p; i++) {
      MPI_Recv(greeting, bufLen, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cout << greeting << endl;
    }
    cout << "Greetings, done!" << endl;
  } else {
    string greeting("Greetings from process " + to_string(id));
    MPI_Send(greeting.c_str(), (int)greeting.size() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}