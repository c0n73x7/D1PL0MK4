exportChart[file_] := (
    Export[
        file,
        Framed[
            RegionPlot[
                -x + 4*y >= 1 && 
                3*x - y >= 1, 
                {x, 0, 4},
                {y, 0, 4},
                FrameLabel -> {Subscript[y, 1], Subscript[y, 2]},
                RotateLabel -> False,
                PlotTheme -> {
                    "Web",
                    "FullAxesGrid",
                    "CoolColor"
                }
            ],
            FrameStyle -> Opacity[0]
        ]
    ]
);

exportChart[StringJoin[{$ScriptCommandLine[[2]], ".svg"}]];
