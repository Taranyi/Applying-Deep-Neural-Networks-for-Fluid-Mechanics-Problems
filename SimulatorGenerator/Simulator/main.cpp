#include "SPlisHSPlasH/Common.h"
#include "Simulator/SimulatorBase.h"

#include "PositionBasedDynamicsWrapper/PBDBoundarySimulator.h"


// Enable memory leak detection
#ifdef _DEBUG
#ifndef EIGEN_ALIGN
	#define new DEBUG_NEW 
#endif
#endif

using namespace SPH;
using namespace std;

SimulatorBase *base = nullptr;


// main 
int main( int argc, char **argv )
{

	srand(time(NULL));

	REPORT_MEMORY_LEAKS;
	base = new SimulatorBase();
	base->init(argc, argv, "SPlisHSPlasH");

	base->run();

	delete base;
	
	return 0;
}

