import java.util.Random;
// You can import more 'standard' Java packages, but no third-party softwares.

/*
 * State IDs:
 * 0 1 2 3 4 5 ... nCols - 1
 * nCols ... nCols * 2 - 1
 * ....
 * nRows ... nRows * nCols - 1
 */

public class RLPractice {
	
	public static final int NUM_ROWS = 2; // Number of rows in the grid world
	public static final int NUM_COLS = 4; // Number of columns in the grid world
	public static final int NUM_ACTIONS = 4; // These correspond to the four possible moves in each cell
	
	public static final double DISCOUNT = 0.9; // Discount factor. You're free to change this.
	public static final double EPSILON = 0.01; // Exploration parameter. You're free to change this.
	public static final double SUCCESS_FACTOR = 0.91; // Probability that
	
	private double P[][][]; // Transition probabilities
	private double R[][];   // Rewards. Assume the rewards are given for each (state, action) pair, unlike the more general formulation of R(s, a, s').
	
	private int nStates = NUM_ROWS * NUM_COLS; // Row-major indexing of the grid cells (see the head comment)
	
	public RLPractice() {
		init();
	}
	
	/*
	 * You can call this method as many times as you want to.
	 * Every call will result in a random re-initialization of the MDP.
	 * You're free to try different options while working on this, but the final score
	 * will be determined by the original version I'm providing.
	 * You don't really have to know what this MDP looks like (after all, that's the premise of RL problems!),
	 * but if you must, it's a simple grid world setting with wrap-around stochastic movements.
	 */
	public void init() {
		Random rnd = new Random();
		P = new double[NUM_ACTIONS][nStates][nStates]; // action-orgState-dstState
		R = new double[NUM_ACTIONS][nStates];
		
		for (int i = 0; i < NUM_ACTIONS; i++) { 
			for (int j = 0; j < nStates; j++) { // from state
				double reward = (rnd.nextInt(10) - 6) * 2;
				int up = j - RLPractice.NUM_COLS;
				if (up < 0)
					up = j + RLPractice.NUM_COLS * (RLPractice.NUM_ROWS - 1);
				int down = j + RLPractice.NUM_COLS;
				if (down >= nStates)
					down = j - RLPractice.NUM_COLS * (RLPractice.NUM_ROWS - 1);
				int curCol = j % RLPractice.NUM_COLS;
				int left = curCol == 0 ? j + NUM_COLS - 1 : j - 1;
				int right = curCol == RLPractice.NUM_COLS - 1 ? j - NUM_COLS + 1 : j + 1;

				R[i][j] = reward; // rewards are independent of actions (for our example)
				for (int k = 0; k < nStates; k++) { // destination state
					switch (i) {
					case 0: // up
						if (k == up)
							P[i][j][k] = RLPractice.SUCCESS_FACTOR;
						else if (k == left || k == right || k == down)
							P[i][j][k] = (1 - RLPractice.SUCCESS_FACTOR) / 3;
						break;
					case 1: // right
						if (k == right)
							P[i][j][k] = RLPractice.SUCCESS_FACTOR;
						else if (k == left || k == up || k == down)
							P[i][j][k] = (1 - RLPractice.SUCCESS_FACTOR) / 3;
						break;
					case 2: // left
						if (k == left)
							P[i][j][k] = RLPractice.SUCCESS_FACTOR;
						else if (k == up || k == right || k == down)
							P[i][j][k] = (1 - RLPractice.SUCCESS_FACTOR) / 3;
						break;
					case 3: // down
						if (k == down)
							P[i][j][k] = RLPractice.SUCCESS_FACTOR;
						else if (k == left || k == right || k == up)
							P[i][j][k] = (1 - RLPractice.SUCCESS_FACTOR) / 3;
						break;
					}
					// P[i][j][k] = 0;
				}
			}
		}

		curState = rnd.nextInt(nStates);
		curStep = 0;
	}

	/**
	 * The following are helper methods that provide interface to the MDP.
	 * You're free to add more helpers, or even change existing ones, as you see fit.
	 */

	public double reward(int state, int action) {
		if(action < 0 || state < 0 || action >= NUM_ACTIONS || state >= nStates) throw new Exception();
		return R[action][state];
	}

	public int getNextState(int state, int action) {
		int[] nextStates = P[action][state];
		int rnd = (new Random()).nextDouble();
		double sofar = 0;
		for(int i = 0; i < nextStates.length; i++) {
			sofar += nextStates[i];
			if(sofar > rnd) return i;
		}
		return nextStates.length - 1;
	}

	public int getNumActions() { return NUM_ACTIONS; }

	public int getNumStates() { return nStates; }


	/**
	 * TODO: Implement the following three methods.
	 */

	public double[] valueIteration() { 		
		// Make sure to print out the max error at each iteration to see that your implementation is converging
		// Return the value table at the end.
	}
	
	public static double[][] qLearning(RLPractice mdp, double gamma, double epsilon) {
		// Implement Q-learning on the current MDP with the given gamma and epsilon hyperparameters.
		// Make sure to print out the max error at each iteration to see that your implementation is converging
		// Return the Q-value table at the end. This table should provide access to the Q-values using the (state, action) indexes.
	}
	
	public static double[][] sarsa(RLPractice mdp, double gamma, double epsilon) {
		// Implement SARSA with epsilon-greedy policy.
		// Make sure to print out the max error at each iteration to see that your implementation is converging
		// Return the Q-value table at the end.
	}

	/*
	 * The main() code is there just to demonstrate a certain usage, and is not mandatory.
	 * You may change the following in any way you want.
	 */

	public static void main(String[] args) {
		RLPractice rl = new RLPractice();
		rl.init();
		System.out.println("Value iteration:");
		double[] vals = rl.valueIteration();
		System.out.println("--------------------------------------------\nQ-learning:");
		double[][] qvals = RLPractice.qLearning(rl, 0.99, 0.05);
		System.out.println("Q(s1, a2) = " + qvals[0][1]);
		System.out.println("--------------------------------------------\nSARSA:");
		qvals = RLPractice.sarsa(rl, 0.99, 0.05);
		System.out.println("Q(s2, a1) = " + qvals[1][0]);
	}
}
