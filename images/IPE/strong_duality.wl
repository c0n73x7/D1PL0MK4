exportChart[file_] := (
    Export[
        file,
        Framed[
            Show[
                Plot[
                    0,
                    {x, -4, 4},
                    PlotStyle -> Darker[Gray],
                    Axes -> {True, False},
                    AxesLabel -> {"R", " "},
                    Ticks -> None
                ],
                Plot[
                    0,
                    {x, -5, 0},
                    PlotTheme -> {
                        "Web",
                        "NoAxes",
                        "CoolColor"
                    }
                ], 
                Plot[
                    0,
                    {x, 0, 5},
                    PlotTheme -> {
                        "Web",
                        "NoAxes",
                        "WarmColor"
                    }
                ],
                PlotRange -> All,
                Epilog -> {
                    Text[b^T y, {2, 0.1}],
                    Text[c^T x, {-2, 0.1}]
                }
            ],
            FrameStyle -> Opacity[0]
        ]
    ]
);

exportChart[StringJoin[{$ScriptCommandLine[[2]], ".svg"}]];
