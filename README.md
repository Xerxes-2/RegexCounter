# RegexCounter: A Expended Solution to HackerRank Problem Count Strings

Original Problem: <https://www.hackerrank.com/challenges/count-strings/problem>

## Features

- Support more than 2 characters in the alphabet
- No explicit bracket needed for unary operator and concatenation
- Support drawing NFA and DFA

### Valid operators

- `|` for union
- `*` for Kleene star
- `+` for Kleene plus
- `?` for optional
- `()` for grouping

### DFA graph

![Alt text](https://g.gravizo.com/source/dfa_graph1?https%3A%2F%2Fraw.githubusercontent.com%2FXerxes%2D2%2FRegexCounter%2Fmain%2FREADME.md)
<details>
<summary></summary>
dfa_graph1
  digraph DFA {
    rankdir=LR;
    node [shape=point]; start;
    node [shape=doublecircle];
    2;
    4;
    6;
    8;
    9;
    10;
    11;
    node [shape=circle];
    start -> 0;
    0 -> 7 [label="a"];
    0 -> 1 [label="b"];
    1 -> 4 [label="a"];
    1 -> 2 [label="b"];
    2 -> 3 [label="a"];
    2 -> 2 [label="b"];
    3 -> 2 [label="b"];
    4 -> 6 [label="a"];
    4 -> 5 [label="b"];
    5 -> 6 [label="a"];
    5 -> 5 [label="b"];
    7 -> 10 [label="a"];
    7 -> 8 [label="b"];
    8 -> 9 [label="a"];
    8 -> 2 [label="b"];
    9 -> 2 [label="b"];
    10 -> 4 [label="a"];
    10 -> 11 [label="b"];
    11 -> 9 [label="a"];
    11 -> 11 [label="b"];
  }
dfa_graph1
</details>
