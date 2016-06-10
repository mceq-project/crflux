import numpy as np
from timeit import default_timer as timer

def kern_CUDA_sparse(nsteps, dX, rho_inv, int_m, dec_m, phi):
    
    calc_precision = np.float32
    from numbapro.cudalib import cusparse
    from numbapro.cudalib.cublas import Blas
    from numbapro import cuda

    cusp = cusparse.Sparse()
    cubl = Blas()
    m, n = int_m.shape
    int_m_nnz = int_m.nnz
    int_m_csrValA = cuda.to_device(int_m.data.astype(calc_precision))
    int_m_csrRowPtrA = cuda.to_device(int_m.indptr)
    int_m_csrColIndA = cuda.to_device(int_m.indices)
    
    dec_m_nnz = dec_m.nnz
    dec_m_csrValA = cuda.to_device(dec_m.data.astype(calc_precision))
    dec_m_csrRowPtrA = cuda.to_device(dec_m.indptr)
    dec_m_csrColIndA = cuda.to_device(dec_m.indices)
    
    cu_curr_phi = cuda.to_device(phi.astype(calc_precision))
    cu_delta_phi = cuda.device_array(phi.shape, dtype=calc_precision)

    descr = cusp.matdescr()
    descr.indexbase = cusparse.CUSPARSE_INDEX_BASE_ZERO
    
    for step in xrange(nsteps):
        if not step % 500:
            print 'Solving for step', step
        cusp.csrmv(trans='N', m=m, n=n, nnz=int_m_nnz,
                   descr=descr,
                   alpha=calc_precision(1.0),
                   csrVal=int_m_csrValA,
                   csrRowPtr=int_m_csrRowPtrA,
                   csrColInd=int_m_csrColIndA,
                   x=cu_curr_phi, beta=calc_precision(0.0), y=cu_delta_phi)
        cusp.csrmv(trans='N', m=m, n=n, nnz=dec_m_nnz,
                   descr=descr,
                   alpha=calc_precision(rho_inv[step]),
                   csrVal=dec_m_csrValA,
                   csrRowPtr=dec_m_csrRowPtrA,
                   csrColInd=dec_m_csrColIndA,
                   x=cu_curr_phi, beta=calc_precision(1.0), y=cu_delta_phi)
        cubl.axpy(alpha=calc_precision(dX[step]), x=cu_delta_phi, y=cu_curr_phi)

if __name__ == '__main__':
    from scipy.sparse import rand as srand
    from scipy.sparse import csr_matrix

    nsteps = 10000
    dX = np.linspace(1,1000,nsteps)
    rho_inv = np.ones_like(dX)
    msize = 1024*8
    int_m = srand(msize, msize, density=0.08, dtype=np.float32)
    dec_m = srand(msize, msize, density=0.01, dtype=np.float32)
    int_m = csr_matrix(int_m)
    dec_m = csr_matrix(dec_m)

    phi = np.random.rand(msize)

    start = timer()
    kern_CUDA_sparse(nsteps, dX, rho_inv, int_m, dec_m, phi)
    print 'Time spent on CUDA stuff', timer() - start
