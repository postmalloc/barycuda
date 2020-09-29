import barycuda.core as bary

def point_in_simplex(pts, n, dim, verts):
  return list(map(bool, bary.point_in_simplex(pts, n, dim, verts)))

def bary_simplex(pts, n, dim, verts):
  return bary.bary_simplex(pts, n, dim, verts)