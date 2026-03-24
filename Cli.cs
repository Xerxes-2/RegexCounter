sealed class AppOptions
{
    public required string Regex { get; init; }
    public required int Length { get; init; }
    public required HashSet<string> DrawTargets { get; init; }
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

        for (var i = 0; i < args.Length; i++)
        {
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
                "Usage: dotnet run -- [--draw nfa,dfa,minDfa] [regex] [length]"
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
        };
    }
}
