//Any change done here would also have to be done in the constants in kernel.cl
#define MAX_DEPTH 10
#define SQRTN 10 //For Grid Points, the grid is SQRTN x SQRTN
#define NUM_POINTS SQRTN*SQRTN
#define USE_HAMM 0 //Use Hammerlsey Points or Grid Points
#define NUM_U 8 //Number of Divisions for theta, any change will change the number of actions 
				//so the size of scatter cdf array in kernel.cl will have to be changed
#define NUM_V 16 //Number of Divisions for phi, this will also change number of actions
#define NUM_ACTIONS NUM_U*NUM_V
#define RL_ON 1 //Use RL
#define LS 0 //Only for the CPU, to use the light sampling method mentioned in Peter Shirley's
			 //Ray Tracing in One Weekend Series