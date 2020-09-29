from ctypes import *
import barycuda as bary

class Point(Structure):
  _fields_ = [
    ('x', c_float),
    ('y', c_float),
    ('z', c_float)
  ]

pts = (Point * 2)()
pts[0] = Point(120.0, 10.0, 5.0);
pts[1] = Point(10.0, 8.0, 9.0);

verts = (Point * 4)()
verts[0] = Point(0.0, 0.0, 0.0)
verts[1] = Point(100.0, 0.0, 0.0)
verts[2] = Point(0.0, 100.0, 0.0)
verts[3] = Point(0.0, 0.0, 100.0)


print(bary.point_in_simplex(pts, 2, 4, verts))

x = bary.bary_simplex(pts, 2, 4, verts);
print(x)
