SetFactory("OpenCASCADE");

//+
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Rectangle(2) = {0, 0, 0, 0.5, 0.5, 0};
//+
Rectangle(3) = {0.5, 0.5, 0, 0.5, 0.5, 0};
//+

BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Surface{3}; Delete; }

//Rectangle(9) = {0, 0.9, 0, -0.5, 0.1, 0};
//Rectangle(10) = {1, 0, 0, 0.5, 0.1, 0};
//+
//BooleanDifference{ Surface{5}; Delete; }{ Surface{9}; Delete; }
//BooleanDifference{ Surface{4}; Delete; }{ Surface{10}; Delete; }

Transfinite Curve {3,4,5,6,8,9,10,11,12} = 200 Using Progression 1;


Transfinite Curve {7,1} = 50 Using Progression 1.05;
Transfinite Curve {8, 2} = 50 Using Progression 1/1.05;

Physical Curve("no_slip", 1) = {3,5,9,12};
//+
Physical Curve("up", 2) = {6,11};
//+
Physical Curve("down", 3) = {4,10};
Physical Surface("voil", 1) = {5,4};
Physical Surface("substance", 2) = {2,3};

Mesh 2;
Mesh.MshFileVersion = 2.2;
Save StrCat(StrPrefix(General.FileName), ".msh");
