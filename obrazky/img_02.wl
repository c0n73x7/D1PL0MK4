exportChart[file_] := (
    Export[
        file,
        Framed[
            Show[
                Graphics[{
                    LightGray,
                    Thick,
                    EdgeForm[{Thick, Gray}],
                    Polygon[{{1, 0}, {0, -1}, {-1, 0}, {0, 1}}]
                }],
                Graphics[{Darker[Gray], PointSize[Large], Point[{1, 0}]}],
                Graphics[{Darker[Gray], PointSize[Large], Point[{0, -1}]}],
                Graphics[{Darker[Gray], PointSize[Large], Point[{-1, 0}]}],
                Graphics[{Darker[Gray], PointSize[Large], Point[{0, 1}]}]
            ],
            FrameStyle -> Opacity[0]
        ]
    ]
);

exportChart[StringJoin[{$ScriptCommandLine[[2]], ".svg"}]];





