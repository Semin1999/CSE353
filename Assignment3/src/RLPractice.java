import java.util.Random;
// You can import more 'standard' Java packages, but no third-party softwares.

/*
 Class(HW): CSE353(HW3)
 Name(ID): Semin Bae(114730530)
 E-mail: semin.bae@stonybrook.edu
 */

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
                double reward = (rnd.nextInt(10) - 6) * 2; // -12 ~ 6 event numbere
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

        // double curState = rnd.nextInt(nStates);
        // double curStep = 0;
    }

    /**
     * The following are helper methods that provide interface to the MDP.
     * You're free to add more helpers, or even change existing ones, as you see fit.
     */

    public double reward(int state, int action) {
        if(action < 0 || state < 0 || action >= NUM_ACTIONS || state >= nStates) {
            System.out.println("Not appropriate parameter(error with parameter value)");
            return 0.0;
        }
        return R[action][state];
    }

    public double getTransitionProbability(int action, int state, int nextState) {
        if(action < 0 || state < 0 || nextState < 0 || action >= NUM_ACTIONS || state >= nStates || nextState >= nStates) {
            System.out.println("Not appropriate parameter(error with parameter value)");
            return 0.0;
        }
        return P[action][state][nextState];
    }

    public int getNextState(int state, int action) {
        double[] nextStates = P[action][state];
        double rnd = (new Random()).nextDouble();
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
        // Initialize iteration limit and convergence tracker
        int epoch = 10000;
        double delta;

        // Initialize V arbitrarily (in this case with zeros)
        double[] valueTable = new double[getNumStates()];  // V(s) = 0 for all s ∈ S+

        // Repeat until convergence or max iterations reached
        for(int i = 0; i < epoch; i++){
            // Delta <- 0
            delta = 0;
            double[] new_valueTable = new double[getNumStates()];

            // For each s ∈ S:
            for(int state = 0; state < getNumStates(); state++){
                // Store old value V(s) implicitly in valueTable[state]

                // Initialize for max_a operation
                double maxValue = Double.NEGATIVE_INFINITY;

                // Compute max_a Σ P^a_ss'[R^a_ss' + γV(s')]
                for(int action = 0; action < getNumActions(); action++) {
                    // Start with immediate reward R^a_ss'
                    double rewardValue = reward(state, action);

                    // Add discounted future rewards: Σ P^a_ss'[γV(s')]
                    for (int primeValue = 0; primeValue < getNumStates(); primeValue++) {
                        double transitionProb = getTransitionProbability(action, state, primeValue);  // P^a_ss'
                        rewardValue += DISCOUNT * transitionProb * valueTable[primeValue];  // γV(s')
                    }

                    // Update max value if current action gives better value
                    if (rewardValue > maxValue)
                        maxValue = rewardValue;
                }

                // V(s) <- max_a Σ P^a_ss'[R^a_ss' + γV(s')]
                new_valueTable[state] = maxValue;

                // Delta <- max(Delta, |v - V(s)|)
                delta = Math.max(delta, Math.abs(new_valueTable[state] - valueTable[state]));
            }

            // Update value table for next iteration
            valueTable = new_valueTable;

            // Print convergence progress
            System.out.printf("\n Value Iteration: %d Iteration: \t Max Error = %f", i, delta);

            // Check if converged (Delta < θ)
            if(delta < EPSILON){
                System.out.printf("\n Converged Value with iteration %d\n", i);
                break;
            }
        }

        return valueTable;
    }


    public static double[][] qLearning(RLPractice mdp, double gamma, double epsilon) {
        // Implement Q-learning on the current MDP with the given gamma and epsilon hyperparameters.
        // Make sure to print out the max error at each iteration to see that your implementation is converging
        // Return the Q-value table at the end. This table should provide access to the Q-values using the (state, action) indexes.

        // initialize num of state and action for visibility
        int nStates = mdp.getNumStates();
        int nActions = mdp.getNumActions();

        // Initialize hyper-parameter
        double learning_rate = 0.1;
        int nEpisode = 100;
        int stepInEpisode = 10000;

        //Initialize Q(s, a) arbitrarily
        double[][] Q = new double[nStates][nActions];

        // Repeat (for each episode):
        for(int episode = 0; episode < nEpisode; episode++){
            //Initialize s
            int s = new Random().nextInt(nStates);

            double maxError = 0;

            // Repeat (for each step of episode) until s is terminal
            for(int step = 0; step < stepInEpisode; step++) {
                // Choose a from s using policy derived from Q (e.g., ε-greedy)
                int a;

                // Random action with probability ε
                if (Math.random() < epsilon) {
                    // System.out.println("야옹");
                    a = new Random().nextInt(nActions);
                }
                // Choose action greedily w.p. 1-ε
                else {
                    a = getGreedyAction(Q, s);
                }

                // Take action a, observe r, s'
                double r = mdp.reward(s, a);
                int sPrime = mdp.getNextState(s, a);

                // Q(s,a) ← Q(s,a) + α[r + γ maxa' Q(s',a') - Q(s,a)]
                // Get maxa' Q(s',a')
                double maxQPrime = Double.NEGATIVE_INFINITY;
                // Choose action greedily for next state (target policy)
                for (int aPrime = 0; aPrime < nActions; aPrime++) {
                    maxQPrime = Math.max(maxQPrime, Q[sPrime][aPrime]);
                }

                // Update Q-value
                double oldQ = Q[s][a];
                Q[s][a] = Q[s][a] + learning_rate * (r + gamma * maxQPrime - Q[s][a]);

                maxError = Math.max(maxError, Math.abs(Q[s][a] - oldQ));

                // s ← s'
                s = sPrime;
            }
            System.out.printf("Q-learning: %d Iteration: \t Max Error = %f\n", episode, maxError);
        }
        return Q;
    }

    public static double[][] sarsa(RLPractice mdp, double gamma, double epsilon) {
        // Implement SARSA with epsilon-greedy policy.
        // Make sure to print out the max error at each iteration to see that your implementation is converging
        // Return the Q-value table at the end.

        // Initialize num of state and action for visibility
        int nStates = mdp.getNumStates();
        int nActions = mdp.getNumActions();

        // Initialize hyper-parameter
        double learning_rate = 0.1;
        int nEpisode = 100;
        int stepInEpisode = 10000;

        //Initialize Q(s, a) arbitrarily
        double[][] Q = new double[nStates][nActions];

        // Repeat (for each episode):
        for (int episode = 0; episode < nEpisode; episode++) {
            //Initialize s
            int s = new Random().nextInt(nStates);

            // Choose a from s using policy derived from Q (e.g., ε-greedy)
            int a;
            if (Math.random() < epsilon) {
                a = new Random().nextInt(nActions);
            } else {
                a = getGreedyAction(Q, s);
            }

            double maxError = 0;

            // Repeat (for each step of episode):
            for (int step = 0; step < stepInEpisode; step++) {
                // Take action a, observe r, s'
                double r = mdp.reward(s, a);
                int sPrime = mdp.getNextState(s, a);

                // Choose a' from s' using policy derived from Q (e.g., ε-greedy)
                int aPrime;
                if (Math.random() < epsilon) {
                    aPrime = new Random().nextInt(nActions);
                } else {
                    aPrime = getGreedyAction(Q, sPrime);
                }

                // Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                double oldQ = Q[s][a];
                Q[s][a] = Q[s][a] + learning_rate * (r + gamma * Q[sPrime][aPrime] - Q[s][a]);

                maxError = Math.max(maxError, Math.abs(Q[s][a] - oldQ));

                // s ← s'; a ← a'
                s = sPrime;
                a = aPrime;
            }

            System.out.printf("SARSA: %d Iteration: \t Max Error = %f\n", episode, maxError);
        }

        return Q;
    }

    // Helper method to get the greedy action (argmax_a Q(s,a))
    private static int getGreedyAction(double[][] Q, int state) {
        int bestAction = 0;
        double bestValue = Q[state][0];

        for(int a = 1; a < Q[state].length; a++) {
            if(Q[state][a] > bestValue) {
                bestValue = Q[state][a];
                bestAction = a;
            }
        }

        return bestAction;
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
        System.out.println("vals[0] = " + vals[0]);
        System.out.println("--------------------------------------------\nQ-learning:");
        double[][] qvals = RLPractice.qLearning(rl, 0.9, 0.01);
        System.out.println("Q(s1, a2) = " + qvals[0][1]);
        System.out.println("--------------------------------------------\nSARSA:");
        qvals = RLPractice.sarsa(rl, 0.9, 0.01);
        System.out.println("Q(s2, a1) = " + qvals[1][0]);
    }
}
