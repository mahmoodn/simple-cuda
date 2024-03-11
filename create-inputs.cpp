#include <fstream>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;

int main(int argc, char** argv)
{
  if (argc != 2) {
    cout << "Usage: ./create_inputs <number_of_elements>\n";
    return 1;
  }
  int N = stoi(argv[1]);
  srand(time(0));
  ofstream f1("inp1.txt");
  ofstream f2("inp2.txt");
  float a = 0.0;
  for (int i = 0; i < N; i++) {
    a = static_cast<float> ( ((rand()%2 == 0) ? (1) : (-1)) * 
                             (rand() % 100000 / 10000.0) );
    f1 << a;
    
    a = static_cast<float> ( ((rand()%2 == 0) ? (1) : (-1)) * 
                             (rand() % 100000 / 10000.0) );
    f2 << a;
    if (i != N-1) {
      f1 << " ";
      f2 << " ";
    }
  }
  f1.close();
  f2.close();
  return 0;
}
