class RegexCountingEngine
{
    public const int MOD = 1000000007;

    public class PipelineResult
    {
        public required NFA Nfa { get; init; }
        public required DFA Dfa { get; init; }
        public required MinDFA MinDfa { get; init; }
        public required int NfaStateCount { get; init; }
        public required int DfaStateCount { get; init; }
        public required int MinDfaStateCount { get; init; }
    }

    public static PipelineResult BuildPipeline(string regexPattern)
    {
        var parser = new Parser(regexPattern);
        var regex = parser.Parse();
        var nfa = regex.ToNFA();
        var dfa = nfa.ToDFA();
        var minDFA = dfa.ToMinimizer().RefineUntilStable().ToMinDFA();

        return new PipelineResult
        {
            Nfa = nfa,
            Dfa = dfa,
            MinDfa = minDFA,
            NfaStateCount = nfa.AllStates().Count,
            DfaStateCount = dfa.AllStates().Count,
            MinDfaStateCount = minDFA.NumberStates().Count,
        };
    }

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
                    result[i][j] = (result[i][j] + matrix1[i][k] * matrix2[k][j] % MOD) % MOD;
                }
            }
        }

        return result;
    }

    public static int CountMatrix(MinDFA minDFA, int length)
    {
        var numbers = minDFA.NumberStates();
        var allStates = numbers.OrderBy(pair => pair.Value).Select(pair => pair.Key).ToArray();
        var acceptStateIndexes = allStates
            .Where(state => state.IsAcceptState)
            .Select(state => numbers[state])
            .ToArray();
        ulong[][] transitionMatrix = new ulong[allStates.Length][];
        for (int i = 0; i < allStates.Length; i++)
        {
            transitionMatrix[i] = new ulong[allStates.Length];
            var state = allStates[i];
            foreach (var symbol in minDFA.Alphabet.Symbols)
            {
                if (state.Transitions.TryGetValue(symbol, out var nextState))
                {
                    transitionMatrix[i][numbers[nextState]]++;
                }
            }
        }

        var resultMatrix = FastMatrixExponentiation(transitionMatrix, length);
        ulong count = 0;
        foreach (var index in acceptStateIndexes)
        {
            count += resultMatrix[0][index];
        }
        return (int)(count % MOD);
    }
}
