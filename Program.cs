using System.Text;

class Alphabet
{
    public HashSet<char> Symbols { get; } = new();

    public static Alphabet operator +(Alphabet a, Alphabet b)
    {
        var result = new Alphabet();
        result.Symbols.UnionWith(a.Symbols);
        result.Symbols.UnionWith(b.Symbols);
        return result;
    }

    public void Add(char symbol)
    {
        Symbols.Add(symbol);
    }
}

class MinDFAState
{
    public bool IsAcceptState { get; set; } = false;
    public Dictionary<char, MinDFAState> Transitions { get; } = new();
    public int Number { get; set; }
}

class MinDFA
{
    public Alphabet Alphabet { get; }
    public MinDFAState StartState { get; }

    public MinDFA(MinDFAState startState, Alphabet alphabet)
    {
        StartState = startState;
        Alphabet = alphabet;
    }

    public HashSet<MinDFAState> AllStates()
    {
        var currentNum = 0;
        var visited = new HashSet<MinDFAState>();
        var stack = new Stack<MinDFAState>();
        stack.Push(StartState);
        while (stack.Count > 0)
        {
            var state = stack.Pop();
            if (!visited.Contains(state))
            {
                visited.Add(state);
                state.Number = currentNum++;
                foreach (var nextState in state.Transitions.Values)
                {
                    stack.Push(nextState);
                }
            }
        }
        return visited;
    }

    // Print the DFA in Graphviz format.
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph DFA {");
        sb.AppendLine("  rankdir=LR;");
        sb.AppendLine("  node [shape=point]; start;");
        sb.AppendLine("  node [shape=doublecircle];");
        foreach (var state in AllStates())
        {
            if (state.IsAcceptState)
            {
                sb.AppendLine($"  {state.Number};");
            }
        }
        sb.AppendLine("  node [shape=circle];");
        sb.AppendLine("  start -> " + StartState.Number + ";");
        foreach (var state in AllStates())
        {
            foreach (var symbol in state.Transitions.Keys)
            {
                sb.AppendLine(
                    $"  {state.Number} -> {state.Transitions[symbol].Number} [label=\"{symbol}\"];"
                );
            }
        }
        sb.AppendLine("}");
        return sb.ToString();
    }
}

class Partition
{
    public Alphabet Alphabet { get; }
    public DFAState StartState { get; }
    public List<HashSet<DFAState>> OldSets { get; set; } = new();
    public Dictionary<DFAState, int> OldMap { get; set; } = new();

    public Partition(
        DFAState startState,
        HashSet<DFAState> nonAcceptStates,
        HashSet<DFAState> acceptStates,
        Alphabet alphabet
    )
    {
        StartState = startState;
        if (nonAcceptStates.Count > 0)
            OldSets.Add(nonAcceptStates);
        if (acceptStates.Count > 0)
            OldSets.Add(acceptStates);
        foreach (var state in nonAcceptStates)
        {
            OldMap[state] = 0;
        }
        foreach (var state in acceptStates)
        {
            OldMap[state] = OldSets.Count - 1;
        }
        Alphabet = alphabet;
    }

    public Partition LoopPartition()
    {
        int oldNum = 0;
        while (oldNum != OldSets.Count)
        {
            oldNum = OldSets.Count;
            NextPartition();
        }
        return this;
    }

    public void NextPartition()
    {
        var newSets = new List<HashSet<DFAState>>();
        var newMap = new Dictionary<DFAState, int>();
        foreach (var set in OldSets)
        {
            var setArray = set.ToArray();
            var newSet = new HashSet<DFAState>() { };
            // find all non-distinguishable states
            var stack = new Stack<int>();
            stack.Push(0);
            while (stack.Count > 0)
            {
                var index = stack.Pop();
                var state1 = setArray[index];
                newSet.Add(state1);
                newMap[state1] = newSets.Count;
                for (var i = index + 1; i < set.Count; i++)
                {
                    var state2 = setArray[i];
                    if (!newSet.Contains(state2) && !Distinguishable(state1, state2))
                    {
                        newSet.Add(state2);
                        newMap[state2] = newSets.Count;
                        stack.Push(i);
                    }
                }
            }
            newSets.Add(newSet);
            // if any remaining states, add to new set
            var remainSet = set.Except(newSet);
            if (remainSet.Any())
            {
                foreach (var state in remainSet)
                {
                    newMap[state] = newSets.Count;
                }
                newSets.Add(remainSet.ToHashSet());
            }
        }
        OldSets = newSets;
        OldMap = newMap;
    }

    public bool Distinguishable(DFAState state1, DFAState state2)
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
            if (OldMap[nextState1] != OldMap[nextState2])
            {
                return true;
            }
        }
        return false;
    }

    public MinDFA ToMinDFA()
    {
        var startState = new MinDFAState();
        var map = new Dictionary<int, MinDFAState>();
        for (var i = 0; i < OldSets.Count; i++)
        {
            if (i == OldMap[StartState])
            {
                map[i] = startState;
            }
            else
            {
                map[i] = new MinDFAState();
            }
        }
        for (var i = 0; i < OldSets.Count; i++)
        {
            var set = OldSets[i];
            var state = map[i];
            foreach (var dfaState in set)
            {
                if (!state.IsAcceptState && dfaState.IsAcceptState)
                {
                    state.IsAcceptState = true;
                }
                foreach (var symbol in Alphabet.Symbols)
                {
                    if (
                        !dfaState.Transitions.ContainsKey(symbol)
                        || state.Transitions.ContainsKey(symbol)
                    )
                    {
                        continue;
                    }
                    var nextState = dfaState.Transitions[symbol];
                    state.Transitions[symbol] = map[OldMap[nextState]];
                }
            }
        }
        return new MinDFA(startState, Alphabet);
    }
}

class DFAState
{
    public bool IsAcceptState { get; set; } = false;
    public Dictionary<char, DFAState> Transitions { get; } = new();

    // A set of NFA states that this DFA state represents.
    public HashSet<NFAState> NFAStates { get; set; } = new();
}

class DFA
{
    public Alphabet Alphabet { get; }
    public DFAState StartState { get; }

    public DFA(DFAState startState, Alphabet alphabet)
    {
        StartState = startState;
        Alphabet = alphabet;
    }

    public HashSet<DFAState> AllStates()
    {
        var visited = new HashSet<DFAState>();
        var stack = new Stack<DFAState>();
        stack.Push(StartState);
        while (stack.Count > 0)
        {
            var state = stack.Pop();
            if (!visited.Contains(state))
            {
                visited.Add(state);
                foreach (var nextState in state.Transitions.Values)
                {
                    stack.Push(nextState);
                }
            }
        }
        return visited;
    }

    // Print the DFA in Graphviz format.
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph DFA {");
        sb.AppendLine("  rankdir=LR;");
        sb.AppendLine("  node [shape=point]; start;");
        sb.AppendLine("  node [shape=doublecircle];");
        foreach (var state in AllStates())
        {
            if (state.IsAcceptState)
            {
                sb.AppendLine($"  {state.GetHashCode()};");
            }
        }
        sb.AppendLine("  node [shape=circle];");
        sb.AppendLine("  start -> " + StartState.GetHashCode() + ";");
        foreach (var state in AllStates())
        {
            foreach (var symbol in state.Transitions.Keys)
            {
                sb.AppendLine(
                    $"  {state.GetHashCode()} -> {state.Transitions[symbol].GetHashCode()} [label=\"{symbol}\"];"
                );
            }
        }
        sb.AppendLine("}");
        return sb.ToString();
    }

    public Partition ToPartition()
    {
        var nonAcceptStates = new HashSet<DFAState>();
        var acceptStates = new HashSet<DFAState>();
        foreach (var state in AllStates())
        {
            if (state.IsAcceptState)
            {
                acceptStates.Add(state);
            }
            else
            {
                nonAcceptStates.Add(state);
            }
        }
        return new Partition(StartState, nonAcceptStates, acceptStates, Alphabet);
    }
}

class NFAState
{
    public Dictionary<char, List<NFAState>> Transitions { get; set; } = new();

    public void AddTransition(char symbol, NFAState state)
    {
        if (!Transitions.TryGetValue(symbol, out var value))
        {
            Transitions[symbol] = value = new List<NFAState>();
        }

        value.Add(state);
    }

    public void AddEpsilonTransition(NFAState state)
    {
        AddTransition('\0', state);
    }
}

class HashSetEqualityComparer<T> : IEqualityComparer<HashSet<T>>
{
    public bool Equals(HashSet<T>? x, HashSet<T>? y)
    {
        if (x == null || y == null)
        {
            return false;
        }
        return x.SetEquals(y);
    }

    public int GetHashCode(HashSet<T>? obj)
    {
        if (obj == null)
        {
            return 0;
        }
        int hash = 0;
        foreach (T t in obj)
        {
            if (t != null)
                hash ^= t.GetHashCode();
        }
        return hash;
    }
}

class NFA
{
    public Alphabet Alphabet { get; }

    public NFAState StartState { get; }
    public NFAState AcceptState { get; }

    public NFA(NFAState startState, NFAState acceptState, Alphabet alphabet)
    {
        StartState = startState;
        AcceptState = acceptState;
        Alphabet = alphabet;
    }

    public DFA ToDFA()
    {
        var dict = new Dictionary<NFAState, HashSet<NFAState>>();
        var map = new Dictionary<HashSet<NFAState>, DFAState>(
            new HashSetEqualityComparer<NFAState>()
        );
        foreach (var state in AllStates())
        {
            dict[state] = EpsilonClosure(state);
        }
        var startState = new DFAState() { NFAStates = dict[StartState] };
        map[startState.NFAStates] = startState;
        var stack = new Stack<DFAState>();
        stack.Push(startState);
        while (stack.Count > 0)
        {
            var state = stack.Pop();
            if (state.NFAStates.Contains(AcceptState))
            {
                state.IsAcceptState = true;
            }
            foreach (var symbol in Alphabet.Symbols)
            {
                var toSet = EpsilonClosure(state.NFAStates, symbol, dict);
                if (toSet.Count > 0)
                {
                    if (!map.TryGetValue(toSet, out var value))
                    {
                        var toState = new DFAState() { NFAStates = toSet };
                        map[toSet] = value = toState;
                        stack.Push(toState);
                    }
                    state.Transitions[symbol] = value;
                }
            }
        }
        return new DFA(startState, Alphabet);
    }

    public static HashSet<NFAState> EpsilonClosure(NFAState initState)
    {
        var closure = new HashSet<NFAState>() { initState };
        var stack = new Stack<NFAState>();
        stack.Push(initState);

        while (stack.Count > 0)
        {
            var state = stack.Pop();
            if (state.Transitions.TryGetValue('\0', out var value))
            {
                value
                    .Where(s => !closure.Contains(s))
                    .ToList()
                    .ForEach(s =>
                    {
                        closure.Add(s);
                        stack.Push(s);
                    });
            }
        }

        return closure;
    }

    public static HashSet<NFAState> EpsilonClosure(
        HashSet<NFAState> fromSet,
        char symbol,
        Dictionary<NFAState, HashSet<NFAState>> dict
    )
    {
        var toStates = new HashSet<NFAState>();
        foreach (var from in fromSet)
        {
            if (from.Transitions.TryGetValue(symbol, out var value))
            {
                value.Where(s => !toStates.Contains(s)).ToList().ForEach(s => toStates.Add(s));
            }
        }
        var toSet = new HashSet<NFAState>();
        foreach (var to in toStates)
        {
            if (dict.TryGetValue(to, out var value))
            {
                toSet.UnionWith(value);
            }
        }
        return toSet;
    }

    public HashSet<NFAState> AllStates()
    {
        var visited = new HashSet<NFAState>();
        var stack = new Stack<NFAState>();
        stack.Push(StartState);
        while (stack.Count > 0)
        {
            var state = stack.Pop();
            if (!visited.Contains(state))
            {
                visited.Add(state);
                foreach (var nextState in state.Transitions.Values)
                {
                    foreach (var next in nextState)
                    {
                        stack.Push(next);
                    }
                }
            }
        }
        return visited;
    }

    // Print the DFA in Graphviz format.
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph DFA {");
        sb.AppendLine("  rankdir=LR;");
        sb.AppendLine("  node [shape=point]; start;");
        sb.AppendLine("  node [shape=doublecircle];");
        foreach (var state in AllStates())
        {
            if (state == AcceptState)
            {
                sb.AppendLine($"  {state.GetHashCode()};");
            }
        }
        sb.AppendLine("  node [shape=circle];");
        sb.AppendLine("  start -> " + StartState.GetHashCode() + ";");
        foreach (var state in AllStates())
        {
            foreach (var symbol in state.Transitions.Keys)
            {
                foreach (var nextState in state.Transitions[symbol])
                {
                    sb.AppendLine(
                        $"  {state.GetHashCode()} -> {nextState.GetHashCode()} [label=\"{symbol}\"];"
                    );
                }
            }
        }
        sb.AppendLine("}");
        return sb.ToString();
    }
}

abstract class Regex
{
    public Alphabet Alphabet { get; set; } = new();

    public abstract override bool Equals(object? obj);
    public abstract override int GetHashCode();

    public NFA ToNFA(NFAState? startState = null, NFAState? acceptState = null)
    {
        startState ??= new NFAState();
        acceptState ??= new NFAState();
        return LinkNFA(startState, acceptState);
    }

    public abstract NFA LinkNFA(NFAState startState, NFAState acceptState);
}

class Literal : Regex
{
    public char Value { get; }

    public Literal(char value)
    {
        Value = value;
        Alphabet.Add(value);
    }

    public override bool Equals(object? obj)
    {
        return obj is Literal other && Value == other.Value;
    }

    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }

    public override NFA LinkNFA(NFAState startState, NFAState acceptState)
    {
        startState.AddTransition(Value, acceptState);
        return new NFA(startState, acceptState, Alphabet);
    }
}

class Concatenation : Regex
{
    public Regex Left { get; }
    public Regex Right { get; }

    public Concatenation(Regex left, Regex right)
    {
        Left = left;
        Right = right;
        Alphabet = left.Alphabet + right.Alphabet;
    }

    public override bool Equals(object? obj)
    {
        return obj is Concatenation other && Left == other.Left && Right == other.Right;
    }

    public override int GetHashCode()
    {
        return Left.GetHashCode() - Right.GetHashCode();
    }

    public override NFA LinkNFA(NFAState startState, NFAState acceptState)
    {
        var leftNFA = Left.ToNFA(startState: startState);
        var middleState = leftNFA.AcceptState;
        var rightNFA = Right.ToNFA(startState: middleState, acceptState: acceptState);
        return new NFA(leftNFA.StartState, rightNFA.AcceptState, Alphabet);
    }
}

class Pipe : Regex
{
    public Regex Left { get; }
    public Regex Right { get; }

    public Pipe(Regex left, Regex right)
    {
        Left = left;
        Right = right;
        Alphabet = left.Alphabet + right.Alphabet;
    }

    public override bool Equals(object? obj)
    {
        return obj is Pipe other && Left == other.Left && Right == other.Right;
    }

    public override int GetHashCode()
    {
        return Left.GetHashCode() + Right.GetHashCode();
    }

    public override NFA LinkNFA(NFAState startState, NFAState acceptState)
    {
        var leftNFA = Left.ToNFA();
        var rightNFA = Right.ToNFA();
        startState.AddEpsilonTransition(leftNFA.StartState);
        startState.AddEpsilonTransition(rightNFA.StartState);
        leftNFA.AcceptState.AddEpsilonTransition(acceptState);
        rightNFA.AcceptState.AddEpsilonTransition(acceptState);
        return new NFA(startState, acceptState, Alphabet);
    }
}

class Star : Regex
{
    public Regex Inner { get; }

    public Star(Regex inner)
    {
        Inner = inner;
        Alphabet = inner.Alphabet;
    }

    public override bool Equals(object? obj)
    {
        return obj is Star other && Inner == other.Inner;
    }

    public override int GetHashCode()
    {
        return Inner.GetHashCode() * '*';
    }

    public override NFA LinkNFA(NFAState startState, NFAState acceptState)
    {
        var innerNFA = Inner.ToNFA();
        startState.AddEpsilonTransition(innerNFA.StartState);
        startState.AddEpsilonTransition(acceptState);
        innerNFA.AcceptState.AddEpsilonTransition(innerNFA.StartState);
        innerNFA.AcceptState.AddEpsilonTransition(acceptState);
        return new NFA(startState, acceptState, Alphabet);
    }
}

class Plus : Regex
{
    public Regex Inner { get; }

    public Plus(Regex inner)
    {
        Inner = inner;
        Alphabet = inner.Alphabet;
    }

    public override bool Equals(object? obj)
    {
        return obj is Plus other && Inner == other.Inner;
    }

    public override int GetHashCode()
    {
        return Inner.GetHashCode() * '+';
    }

    public override NFA LinkNFA(NFAState startState, NFAState acceptState)
    {
        var innerNFA = Inner.ToNFA();
        startState.AddEpsilonTransition(innerNFA.StartState);
        innerNFA.AcceptState.AddEpsilonTransition(innerNFA.StartState);
        innerNFA.AcceptState.AddEpsilonTransition(acceptState);
        return new NFA(startState, acceptState, Alphabet);
    }
}

class QMark : Regex
{
    public Regex Inner { get; }

    public QMark(Regex inner)
    {
        Inner = inner;
        Alphabet = inner.Alphabet;
    }

    public override bool Equals(object? obj)
    {
        return obj is QMark other && Inner == other.Inner;
    }

    public override int GetHashCode()
    {
        return Inner.GetHashCode() * '?';
    }

    public override NFA LinkNFA(NFAState startState, NFAState acceptState)
    {
        var innerNFA = Inner.ToNFA();
        startState.AddEpsilonTransition(acceptState);
        startState.AddEpsilonTransition(innerNFA.StartState);
        innerNFA.AcceptState.AddEpsilonTransition(acceptState);
        return new NFA(startState, acceptState, Alphabet);
    }
}

class Parser
{
    private readonly string r;
    private int i = 0;
    private char Cur => r.Length > i ? r[i] : '\0';

    public Parser(string r)
    {
        this.r = r;
    }

    private void Match(char ch)
    {
        if (Cur == ch)
        {
            i++;
        }
        else
        {
            throw new Exception($"Expected {ch} at position {i}");
        }
    }

    public Regex ParseLiteral()
    {
        var ch = Cur;
        i++;
        return new Literal(ch);
    }

    public Regex ParseParen()
    {
        Match('(');
        var regex = ParsePipe();
        Match(')');
        return regex;
    }

    public Regex ParseUnary()
    {
        var regex = Cur == '(' ? ParseParen() : ParseLiteral();
        while (Cur == '*' || Cur == '+' || Cur == '?')
        {
            switch (Cur)
            {
                case '*':
                    regex = new Star(regex);
                    break;
                case '+':
                    regex = new Plus(regex);
                    break;
                case '?':
                    regex = new QMark(regex);
                    break;
            }
            i++;
        }
        return regex;
    }

    public Regex ParseConcatenation()
    {
        var left = ParseUnary();
        if (Cur == '|' || Cur == ')' || Cur == '\0')
        {
            return left;
        }
        var right = ParseConcatenation();
        return new Concatenation(left, right);
    }

    public Regex ParsePipe()
    {
        var left = ParseConcatenation();
        if (Cur != '|')
        {
            return left;
        }
        i++;
        var right = ParsePipe();
        return new Pipe(left, right);
    }
}

class Result
{
    public const int MOD = 1000000007;

    public static ulong[][] FastMatrixExponentiation(ulong[][] baseMatrix, int power)
    {
        int size = baseMatrix.Length;
        ulong[][] resultMatrix = GetIdentityMatrix(size);

        while (power > 0)
        {
            if ((power & 1) == 1)
            {
                resultMatrix = MultiplyMatrices(resultMatrix, baseMatrix);
            }
            baseMatrix = MultiplyMatrices(baseMatrix, baseMatrix);
            power >>= 1;
        }

        return resultMatrix;
    }

    public static ulong[][] GetIdentityMatrix(int size)
    {
        ulong[][] identity = new ulong[size][];

        for (int i = 0; i < size; i++)
        {
            identity[i] = new ulong[size];
            identity[i][i] = 1;
        }

        return identity;
    }

    public static ulong[][] MultiplyMatrices(ulong[][] matrix1, ulong[][] matrix2)
    {
        int size = matrix1.Length;
        ulong[][] result = new ulong[size][];

        for (int i = 0; i < size; i++)
        {
            result[i] = new ulong[size];
            for (int j = 0; j < size; j++)
            {
                for (int k = 0; k < size; k++)
                {
                    result[i][j] = (result[i][j] + (matrix1[i][k] * matrix2[k][j]) % MOD) % MOD;
                }
            }
        }

        return result;
    }

    public static int CountMatrix(MinDFA minDFA, int l)
    {
        // Create transition matrix for the DFA
        var allStates = minDFA.AllStates().OrderBy(s => s.Number).ToArray();
        var acceptStateIndexes = allStates
            .Where(s => s.IsAcceptState)
            .Select(s => s.Number)
            .ToArray();
        ulong[][] transitionMatrix = new ulong[allStates.Length][];
        for (int i = 0; i < allStates.Length; i++)
        {
            transitionMatrix[i] = new ulong[allStates.Length];
            var state = allStates[i];
            foreach (var symbol in minDFA.Alphabet.Symbols)
            {
                if (state.Transitions.TryGetValue(symbol, out var value))
                {
                    transitionMatrix[i][value.Number]++;
                }
            }
        }
        // exponentiate the matrix
        var resultMatrix = FastMatrixExponentiation(transitionMatrix, l);
        // count the number of strings
        ulong count = 0;
        foreach (var index in acceptStateIndexes)
        {
            count += resultMatrix[0][index];
        }
        return (int)(count % MOD);
    }

    public static int CountStrings(string r, int l)
    {
        var parser = new Parser(r);
        var regex = parser.ParsePipe();
        var nfa = regex.ToNFA();
        var dfa = nfa.ToDFA();
        var minDFA = dfa.ToPartition().LoopPartition().ToMinDFA();
        File.WriteAllText(@"dfa.dot", minDFA.ToString());
        return CountMatrix(minDFA, l);
    }
}

class Solution
{
    public static void Main()
    {
        var result = Result.CountStrings(
            "((((((adc|sss))))))",
            10
        );
        Console.WriteLine(result);
    }
}
