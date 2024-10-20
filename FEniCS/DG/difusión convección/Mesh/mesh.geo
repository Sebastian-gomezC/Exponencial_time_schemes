SetFactory("OpenCASCADE");

//+
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Rectangle(2) = {0, 0, 0, 0.5, 0.5, 0};
//+
Rectangle(3) = {0.5, 0.5, 0, 0.5, 0.5, 0};
//+

BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Surface{3}; Delete; }

Rectangle(9) = {0, 0.9, 0, -0.5, 0.1, 0};
Rectangle(10) = {1, 0, 0, 0.5, 0.1, 0};
//+
BooleanDifference{ Surface{5}; Delete; }{ Surface{9}; Delete; }
BooleanDifference{ Surface{4}; Delete; }{ Surface{10}; Delete; }

Transfinite Curve {9,10,11,12,23,26} = 50 Using Progression 1;
Transfinite Curve {22,25} = 40 Using Progression 1;
Transfinite Curve {24,21} = 10 Using Progression 1;

Transfinite Curve {7,1} = 50 Using Progression 1;
Transfinite Curve {8, 2} = 50 Using Progression 1/1;

Physical Curve("no_slip", 1) = {9, 22, 10, 26, 25, 12, 11, 23};
//+
Physical Curve("in", 2) = {21};
//+
Physical Curve("out", 3) = {24};
Physical Surface("voil", 1) = {5,4};
Physical Surface("substance", 2) = {2,3};

Mesh 2;
Mesh.MshFileVersion = 2.2;
Save StrCat(StrPrefix(General.FileName), ".msh");
