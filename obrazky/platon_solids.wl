exportChart[file_] := (
    Export[
        file,
        Framed[
            Graphics3D[{
                Opacity[0.8],
                PolyhedronData[
                    "Icosahedron",
                    "BoundaryMeshRegion"
                ]},
                Boxed -> False
            ]
        ]
    ]
);

exportChart[StringJoin[{$ScriptCommandLine[[2]], ".svg"}]];
