sealed class AppOptions
{
    public required string Regex { get; init; }
    public required int Length { get; init; }
    public required HashSet<string> DrawTargets { get; init; }
    public required bool Benchmark { get; init; }
    public required int BenchmarkIterations { get; init; }
}

static class CommandLine
{
    public static AppOptions Parse(
        string[] args,
        string defaultRegex,
        int defaultLength,
        IEnumerable<string> validDrawTargets
    )
    {
        string regex = defaultRegex;
        int length = defaultLength;
        var validTargets = new HashSet<string>(validDrawTargets, StringComparer.OrdinalIgnoreCase);
        var drawTargets = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var positionals = new List<string>();
        var benchmark = false;
        var benchmarkIterations = 20;

        for (var i = 0; i < args.Length; i++)
        {
            if (string.Equals(args[i], "--benchmark", StringComparison.OrdinalIgnoreCase))
            {
                benchmark = true;
                continue;
            }

            if (string.Equals(args[i], "--benchmark-iterations", StringComparison.OrdinalIgnoreCase))
            {
                if (i + 1 >= args.Length || !int.TryParse(args[++i], out benchmarkIterations))
                {
                    throw new ArgumentException("Missing or invalid value for --benchmark-iterations.");
                }
                if (benchmarkIterations <= 0)
                {
                    throw new ArgumentException("--benchmark-iterations must be positive.");
                }
                continue;
            }

            if (string.Equals(args[i], "--draw", StringComparison.OrdinalIgnoreCase))
            {
                if (i + 1 >= args.Length)
                {
                    throw new ArgumentException("Missing value for --draw.");
                }

                foreach (
                    var target in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                )
                {
                    if (!validTargets.Contains(target))
                    {
                        throw new ArgumentException(
                            $"Unknown draw target '{target}'. Valid values: {string.Join(", ", validTargets)}."
                        );
                    }

                    drawTargets.Add(target);
                }

                continue;
            }

            positionals.Add(args[i]);
        }

        if (positionals.Count > 2)
        {
            throw new ArgumentException(
                "Usage: dotnet run -- [--draw nfa,dfa,minDfa] [--benchmark] [--benchmark-iterations N] [regex] [length]"
            );
        }

        if (positionals.Count >= 1)
        {
            regex = positionals[0];
        }

        if (positionals.Count == 2 && !int.TryParse(positionals[1], out length))
        {
            throw new ArgumentException($"Invalid length '{positionals[1]}'.");
        }

        return new AppOptions
        {
            Regex = regex,
            Length = length,
            DrawTargets = drawTargets,
            Benchmark = benchmark,
            BenchmarkIterations = benchmarkIterations,
        };
    }
}
