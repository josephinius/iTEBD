# iTEBD
  This code is an implementation of the iTEBD algorithm based on the article: 
 
  Classical simulation of infinite-size quantum lattice systems in one spatial dimension
  Phys. Rev. Lett 98, 070201 (2007), http://arxiv.org/abs/cond-mat/0605597
 
  For linear algebra (Singular Value Decomposition (SVD) in particular), 
  Eigen library (3.2.7) is called. 
 
  Compilation under Mac:
 
  g++ -m64 -O3 -I/.../Eigen main.cpp -o main.x
