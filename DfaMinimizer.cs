class DfaMinimizer
{
    public Alphabet Alphabet { get; }
    public DFAState StartState { get; }
    public List<HashSet<DFAState>> Partitions { get; set; } = [];
    public Dictionary<DFAState, int> PartitionMap { get; set; } = [];

    public DfaMinimizer(
        DFAState startState,
        HashSet<DFAState> nonAcceptStates,
        HashSet<DFAState> acceptStates,
        Alphabet alphabet
    )
    {
        StartState = startState;
        if (nonAcceptStates.Count > 0)
            Partitions.Add(nonAcceptStates);
        if (acceptStates.Count > 0)
            Partitions.Add(acceptStates);
        foreach (var state in nonAcceptStates)
        {
            PartitionMap[state] = 0;
        }
        foreach (var state in acceptStates)
        {
            PartitionMap[state] = Partitions.Count - 1;
        }
        Alphabet = alphabet;
    }

    public DfaMinimizer RefineUntilStable()
    {
        int previousPartitionCount = 0;
        while (previousPartitionCount != Partitions.Count)
        {
            previousPartitionCount = Partitions.Count;
            RefineOnce();
        }
        return this;
    }

    public void RefineOnce()
    {
        var newPartitions = new List<HashSet<DFAState>>();
        var newPartitionMap = new Dictionary<DFAState, int>();
        foreach (var partition in Partitions)
        {
            var partitionArray = partition.ToArray();
            var equivalentStates = new HashSet<DFAState>();
            var stack = new Stack<int>();
            stack.Push(0);
            while (stack.Count > 0)
            {
                var index = stack.Pop();
                var state1 = partitionArray[index];
                equivalentStates.Add(state1);
                newPartitionMap[state1] = newPartitions.Count;
                for (var i = index + 1; i < partition.Count; i++)
                {
                    var state2 = partitionArray[i];
                    if (!equivalentStates.Contains(state2) && !AreDistinguishable(state1, state2))
                    {
                        equivalentStates.Add(state2);
                        newPartitionMap[state2] = newPartitions.Count;
                        stack.Push(i);
                    }
                }
            }
            newPartitions.Add(equivalentStates);
            var remainingStates = partition.Except(equivalentStates);
            if (remainingStates.Any())
            {
                foreach (var state in remainingStates)
                {
                    newPartitionMap[state] = newPartitions.Count;
                }
                newPartitions.Add([.. remainingStates]);
            }
        }
        Partitions = newPartitions;
        PartitionMap = newPartitionMap;
    }

    public bool AreDistinguishable(DFAState state1, DFAState state2)
    {
        foreach (var symbol in Alphabet.Symbols)
        {
            if (!state1.Transitions.ContainsKey(symbol) && !state2.Transitions.ContainsKey(symbol))
            {
                continue;
            }
            if (
                !state1.Transitions.TryGetValue(symbol, out var nextState1)
                || !state2.Transitions.TryGetValue(symbol, out var nextState2)
            )
            {
                return true;
            }
            if (PartitionMap[nextState1] != PartitionMap[nextState2])
            {
                return true;
            }
        }
        return false;
    }

    public MinDFA ToMinDFA()
    {
        var startState = new DFAState();
        var stateMap = new Dictionary<int, DFAState>();
        for (var i = 0; i < Partitions.Count; i++)
        {
            stateMap[i] = i == PartitionMap[StartState] ? startState : new DFAState();
        }

        for (var i = 0; i < Partitions.Count; i++)
        {
            var partition = Partitions[i];
            var minimizedState = stateMap[i];
            foreach (var dfaState in partition)
            {
                if (!minimizedState.IsAcceptState && dfaState.IsAcceptState)
                {
                    minimizedState.IsAcceptState = true;
                }
                foreach (var symbol in Alphabet.Symbols)
                {
                    if (
                        !dfaState.Transitions.ContainsKey(symbol)
                        || minimizedState.Transitions.ContainsKey(symbol)
                    )
                    {
                        continue;
                    }
                    var nextState = dfaState.Transitions[symbol];
                    minimizedState.Transitions[symbol] = stateMap[PartitionMap[nextState]];
                }
            }
        }

        return new MinDFA(startState, Alphabet);
    }
}
