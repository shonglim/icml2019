interface Task3 {
    public EuState getInitialState();
    public int numActions();
    public boolean isGoal(EuState s);
    public Pair<Double,EuState> nextState(EuState s,int a);
    public Task3 copy(long seed);
    public EuState extract_feature(EuState s,int feat_type);
    public void set_noise(double noise);
}
