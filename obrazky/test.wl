exportChart[file_] := (
    Export[
        file,
        Framed[
            RegionPlot[
                -x + 3*y <= 4 && 
                4*x - y <= 6, 
                {x, 0, 2.5}, 
                {y, 0, 2.5},
                FrameLabel -> {Subscript[x, 1], Subscript[x, 2]},
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
