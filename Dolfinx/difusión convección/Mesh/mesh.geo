SetFactory("OpenCASCADE");

Rectangle(1) = {0, 0, 0, 1, 1, 0};

Transfinite Curve {1,2,3,4} = 4 Using Progression 1;

Transfinite Surface {1} ;

Physical Curve("up", 2) = {3};

Physical Curve("down", 3) = {1};
Physical Surface("vol", 1) = {1};

Mesh 2;
Mesh.MshFileVersion = 2.2;
Save StrCat(StrPrefix(General.FileName), ".msh");
