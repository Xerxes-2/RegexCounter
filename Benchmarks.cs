using System.Diagnostics;

class BenchmarkResult
{
    public required string Name { get; init; }
    public required int Count { get; init; }
    public required double ElapsedMilliseconds { get; init; }
    public required int Iterations { get; init; }

    public double AverageMilliseconds => ElapsedMilliseconds / Iterations;
}

static class RegexCountingBenchmarks
{
    public static BenchmarkResult BenchmarkDfaEndToEnd(
        string regexPattern,
        int length,
        int iterations
    )
    {
        return Benchmark(
            "DFA E2E",
            () =>
            {
                var parser = new Parser(regexPattern);
                var regex = parser.Parse();
                var nfa = regex.ToNFA();
                var dfa = nfa.ToDFA();
                return RegexCountingEngine.CountMatrix(dfa, length);
            },
            iterations
        );
    }

    public static BenchmarkResult BenchmarkMinDfaEndToEnd(
        string regexPattern,
        int length,
        int iterations
    )
    {
        return Benchmark(
            "MinDFA E2E",
            () =>
            {
                var pipeline = RegexCountingEngine.BuildPipeline(regexPattern);
                return RegexCountingEngine.CountMatrix(pipeline.MinDfa, length);
            },
            iterations
        );
    }

    private static BenchmarkResult Benchmark(string name, Func<int> action, int iterations)
    {
        action();

        int count = 0;
        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < iterations; i++)
        {
            count = action();
        }
        stopwatch.Stop();

        return new BenchmarkResult
        {
            Name = name,
            Count = count,
            ElapsedMilliseconds = stopwatch.Elapsed.TotalMilliseconds,
            Iterations = iterations,
        };
    }
}
