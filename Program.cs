class Alphabet
{
    public HashSet<char> Symbols { get; } = [];

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

class MinDFA(DFAState startState, Alphabet alphabet)
{
    public Alphabet Alphabet { get; } = alphabet;
    public DFAState StartState { get; } = startState;

    public Dictionary<DFAState, int> NumberStates()
    {
        return GraphTraversal.NumberStates(StartState, state => state.Transitions.Values);
    }

    // Print the DFA in Graphviz format.
    public override string ToString()
    {
        var numbers = NumberStates();
        return DotRenderer.Render(
            "DFA",
            StartState,
            numbers,
            state => state.IsAcceptState,
            state => state.Transitions.Select(transition => (transition.Key, transition.Value))
        );
    }
}

class DFAState
{
    public bool IsAcceptState { get; set; } = false;
    public Dictionary<char, DFAState> Transitions { get; } = [];

    // A set of NFA states that this DFA state represents.
    public HashSet<NFAState> NFAStates { get; set; } = [];
}

class DFA(DFAState startState, Alphabet alphabet)
{
    public Alphabet Alphabet { get; } = alphabet;
    public DFAState StartState { get; } = startState;

    public HashSet<DFAState> AllStates()
    {
        return GraphTraversal.CollectStates(StartState, state => state.Transitions.Values);
    }

    public Dictionary<DFAState, int> NumberStates()
    {
        return GraphTraversal.NumberStates(StartState, state => state.Transitions.Values);
    }

    // Print the DFA in Graphviz format.
    public override string ToString()
    {
        var numbers = NumberStates();
        return DotRenderer.Render(
            "DFA",
            StartState,
            numbers,
            state => state.IsAcceptState,
            state => state.Transitions.Select(transition => (transition.Key, transition.Value))
        );
    }

    public DfaMinimizer ToMinimizer()
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
        return new DfaMinimizer(StartState, nonAcceptStates, acceptStates, Alphabet);
    }
}

class NFAState
{
    public Dictionary<char, List<NFAState>> Transitions { get; set; } = [];

    public void AddTransition(char symbol, NFAState state)
    {
        if (!Transitions.TryGetValue(symbol, out var value))
        {
            Transitions[symbol] = value = [];
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

class NFA(NFAState startState, NFAState acceptState, Alphabet alphabet)
{
    public Alphabet Alphabet { get; } = alphabet;

    public NFAState StartState { get; } = startState;
    public NFAState AcceptState { get; } = acceptState;

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
        return GraphTraversal.CollectStates(
            StartState,
            state => state.Transitions.Values.SelectMany(nextStates => nextStates)
        );
    }

    public Dictionary<NFAState, int> NumberStates()
    {
        return GraphTraversal.NumberStates(
            StartState,
            state => state.Transitions.Values.SelectMany(nextStates => nextStates)
        );
    }

    // Print the DFA in Graphviz format.
    public override string ToString()
    {
        var numbers = NumberStates();
        return DotRenderer.Render(
            "NFA",
            StartState,
            numbers,
            state => state == AcceptState,
            state =>
                state.Transitions.SelectMany(
                    transition => transition.Value.Select(nextState => (transition.Key, nextState))
                )
        );
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

abstract class UnaryRegex : Regex
{
    protected UnaryRegex(Regex inner)
    {
        Inner = inner;
        Alphabet = inner.Alphabet;
    }

    public Regex Inner { get; }

    protected bool HasSameInner(UnaryRegex other)
    {
        return Inner == other.Inner;
    }
}

abstract class BinaryRegex : Regex
{
    protected BinaryRegex(Regex left, Regex right)
    {
        Left = left;
        Right = right;
        Alphabet = left.Alphabet + right.Alphabet;
    }

    public Regex Left { get; }
    public Regex Right { get; }

    protected bool HasSameOperands(BinaryRegex other)
    {
        return Left == other.Left && Right == other.Right;
    }
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

class Concatenation(Regex left, Regex right) : BinaryRegex(left, right)
{
    public override bool Equals(object? obj)
    {
        return obj is Concatenation other && HasSameOperands(other);
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

class Pipe(Regex left, Regex right) : BinaryRegex(left, right)
{
    public override bool Equals(object? obj)
    {
        return obj is Pipe other && HasSameOperands(other);
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

class Star(Regex inner) : UnaryRegex(inner)
{
    public override bool Equals(object? obj)
    {
        return obj is Star other && HasSameInner(other);
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

class Plus(Regex inner) : UnaryRegex(inner)
{
    public override bool Equals(object? obj)
    {
        return obj is Plus other && HasSameInner(other);
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

class QMark(Regex inner) : UnaryRegex(inner)
{
    public override bool Equals(object? obj)
    {
        return obj is QMark other && HasSameInner(other);
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

class Parser(string r)
{
    private readonly string r = r;
    private int i = 0;
    private char Cur => r.Length > i ? r[i] : '\0';

    private static readonly Dictionary<char, char[]> MetaCharacters = new()
    {
        ['d'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        ['l'] = Enumerable.Range('a', 26).Select(value => (char)value).ToArray(),
        ['u'] = Enumerable.Range('A', 26).Select(value => (char)value).ToArray(),
        ['w'] = Enumerable
            .Range('a', 26)
            .Concat(Enumerable.Range('A', 26))
            .Concat(Enumerable.Range('0', 10))
            .Select(value => (char)value)
            .Append('_')
            .ToArray(),
        ['s'] = [' ', '\t', '\n', '\r'],
    };

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

    public Regex Parse()
    {
        var regex = ParsePipe();
        if (Cur != '\0')
        {
            throw new Exception($"Unexpected token '{Cur}' at position {i}");
        }
        return regex;
    }

    public Regex ParseAtom()
    {
        if (Cur == '(')
        {
            return ParseParen();
        }

        return ParseLiteralOrMeta();
    }

    public Regex ParseLiteralOrMeta()
    {
        if (Cur == '\0')
        {
            throw new Exception($"Unexpected end of input at position {i}");
        }

        if (Cur == '\\')
        {
            return ParseEscape();
        }

        if (Cur is '|' or ')' or '*' or '+' or '?')
        {
            throw new Exception($"Unexpected token '{Cur}' at position {i}");
        }

        var ch = Cur;
        i++;
        return new Literal(ch);
    }

    private Regex ParseEscape()
    {
        Match('\\');
        if (Cur == '\0')
        {
            throw new Exception($"Unexpected end of input at position {i}");
        }

        var escapedChar = Cur;
        i++;

        if (MetaCharacters.TryGetValue(escapedChar, out var expandedSymbols))
        {
            return BuildUnion(expandedSymbols);
        }

        return new Literal(escapedChar);
    }

    private static Regex BuildUnion(IReadOnlyList<char> symbols)
    {
        if (symbols.Count == 0)
        {
            throw new ArgumentException("Meta character expansion cannot be empty.");
        }

        Regex regex = new Literal(symbols[0]);
        for (var j = 1; j < symbols.Count; j++)
        {
            regex = new Pipe(regex, new Literal(symbols[j]));
        }
        return regex;
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
        var regex = ParseAtom();
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

class Solution
{
    private static readonly string DefaultRegex =
        "(((a(ba))(b*))|(((b|a)|(aa))((b((b|((b*)|(((((b*)a)b)*)*)))*))|(((a(b*))a)|(b|a)))))";
    private const int DefaultLength = 43625841;

    private static readonly Dictionary<string, Action<RegexCountingEngine.PipelineResult>> DrawActions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["nfa"] = pipeline => File.WriteAllText("nfa.dot", pipeline.Nfa.ToString()),
            ["dfa"] = pipeline => File.WriteAllText("dfa.dot", pipeline.Dfa.ToString()),
            ["mindfa"] = pipeline => File.WriteAllText("minDfa.dot", pipeline.MinDfa.ToString()),
        };

    public static void Main()
    {
        try
        {
            var options = CommandLine.Parse(
                Environment.GetCommandLineArgs()[1..],
                DefaultRegex,
                DefaultLength,
                DrawActions.Keys
            );
            var stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();
            var pipeline = RegexCountingEngine.BuildPipeline(options.Regex);
            foreach (var drawTarget in options.DrawTargets)
            {
                DrawActions[drawTarget](pipeline);
            }
            var result = RegexCountingEngine.CountMatrix(pipeline.MinDfa, options.Length);
            stopwatch.Stop();
            Console.WriteLine($"ElapsedMs: {stopwatch.ElapsedMilliseconds}");
            Console.WriteLine($"Count: {result}");
            Console.WriteLine(
                $"States: NFA={pipeline.NfaStateCount}, DFA={pipeline.DfaStateCount}, MinDFA={pipeline.MinDfaStateCount}"
            );
            if (options.Benchmark)
            {
                var dfaBenchmark = RegexCountingBenchmarks.BenchmarkDfaEndToEnd(
                    options.Regex,
                    options.Length,
                    options.BenchmarkIterations
                );
                var minDfaBenchmark = RegexCountingBenchmarks.BenchmarkMinDfaEndToEnd(
                    options.Regex,
                    options.Length,
                    options.BenchmarkIterations
                );
                Console.WriteLine(
                    $"BenchmarkIterations: {options.BenchmarkIterations}"
                );
                Console.WriteLine(
                    $"Benchmark DFA E2E: count={dfaBenchmark.Count}, totalMs={dfaBenchmark.ElapsedMilliseconds:F3}, avgMs={dfaBenchmark.AverageMilliseconds:F4}"
                );
                Console.WriteLine(
                    $"Benchmark MinDFA E2E: count={minDfaBenchmark.Count}, totalMs={minDfaBenchmark.ElapsedMilliseconds:F3}, avgMs={minDfaBenchmark.AverageMilliseconds:F4}"
                );
                if (minDfaBenchmark.ElapsedMilliseconds > 0)
                {
                    Console.WriteLine(
                        $"Benchmark E2E Speedup: {dfaBenchmark.ElapsedMilliseconds / minDfaBenchmark.ElapsedMilliseconds:F2}x"
                    );
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            Environment.ExitCode = 1;
        }
    }
}
