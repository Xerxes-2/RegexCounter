using System.Text;

static class GraphTraversal
{
    public static HashSet<TState> CollectStates<TState>(
        TState startState,
        Func<TState, IEnumerable<TState>> getNeighbors
    )
        where TState : notnull
    {
        var visited = new HashSet<TState>();
        var stack = new Stack<TState>();
        stack.Push(startState);
        while (stack.Count > 0)
        {
            var state = stack.Pop();
            if (visited.Add(state))
            {
                foreach (var nextState in getNeighbors(state))
                {
                    stack.Push(nextState);
                }
            }
        }
        return visited;
    }

    public static Dictionary<TState, int> NumberStates<TState>(
        TState startState,
        Func<TState, IEnumerable<TState>> getNeighbors
    )
        where TState : notnull
    {
        var currentNum = 0;
        var numbers = new Dictionary<TState, int>();
        var stack = new Stack<TState>();
        stack.Push(startState);
        while (stack.Count > 0)
        {
            var state = stack.Pop();
            if (!numbers.ContainsKey(state))
            {
                numbers[state] = currentNum++;
                foreach (var nextState in getNeighbors(state))
                {
                    stack.Push(nextState);
                }
            }
        }
        return numbers;
    }
}

static class DotRenderer
{
    public static string Render<TState>(
        string graphName,
        TState startState,
        IReadOnlyDictionary<TState, int> numbers,
        Func<TState, bool> isAcceptState,
        Func<TState, IEnumerable<(char Symbol, TState NextState)>> getTransitions
    )
        where TState : notnull
    {
        var sb = new StringBuilder();
        sb.AppendLine($"digraph {graphName} {{");
        sb.AppendLine("  rankdir=LR;");
        sb.AppendLine("  node [shape=point]; start;");
        sb.AppendLine("  node [shape=doublecircle];");
        foreach (var state in numbers.Keys)
        {
            if (isAcceptState(state))
            {
                sb.AppendLine($"  {numbers[state]};");
            }
        }
        sb.AppendLine("  node [shape=circle];");
        sb.AppendLine($"  start -> {numbers[startState]};");
        foreach (var state in numbers.Keys)
        {
            foreach (var (symbol, nextState) in getTransitions(state))
            {
                sb.AppendLine(
                    $"  {numbers[state]} -> {numbers[nextState]} [label=\"{FormatLabel(symbol)}\"];"
                );
            }
        }
        sb.AppendLine("}");
        return sb.ToString();
    }

    private static string FormatLabel(char symbol)
    {
        return symbol switch
        {
            '\0' => "eps",
            '"' => "\\\"",
            '\\' => "\\\\",
            _ => symbol.ToString(),
        };
    }
}
