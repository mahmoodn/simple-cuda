#include <fstream>
#include <math.h>
#include <time.h>

#define N 65636

using namespace std;

int main()
{
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
