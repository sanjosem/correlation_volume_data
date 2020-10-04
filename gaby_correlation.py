class gaby_correlation:

    def __init__(self,datafile,meshfile,iref_probe):
        import os
        import numpy
        
        if type(datafile) is list:
            nfiles = len(datafile)
            for filename in datafile:
                if not os.path.isfile(filename):
                    raise OSError('Data File {0:s} can not be found'.format(filename))

            self.filename = datafile
            self.multi = True
            
        elif type(datafile) is str:
            if os.path.isfile(datafile):
                self.filename = datafile
                self.multi = False
            else:
                raise OSError('Data File {0:s} can not be found'.format(datafile))
                
        else:
            raise TypeError('datafile is not of proper format')
        
        
        if os.path.isfile(meshfile):
            self.mesh = meshfile
        else:
            raise OSError('Mesh File {0:s} can not be found'.format(meshfile))
        
        if type(iref_probe) is int:
            self.iref_probe = iref_probe
        else:
            raise TypeError('integer is expected for iref_probe')
            
        self.coords = None
        self.dims = None
        self.fid = None
        
    def open_file(self,ifile=0):
        import h5py

        if self.fid is None:
            if self.multi:
                self.fid = h5py.File(self.filename[ifile],'r')
            else:
                self.fid = h5py.File(self.filename,'r')
        else:
            self.close_file()
            self.open_file(ifile)
        
    def close_file(self):
        if self.fid is None:
            print('File {0:s} is not open'.format(self.filename))
        else:
            self.fid = self.fid.close()
            self.fid = None
    
    
    def read_coordinates(self):
        import h5py 
        
        if self.coords is None:
            self.coords = dict()
            
        fm = h5py.File(self.mesh,'r')
        self.coords['x'] = fm['x'][()]
        self.coords['y'] = fm['y'][()]
        self.coords['z'] = fm['z'][()]
        fm.close()
        
        self.dims = self.coords['x'].shape
        
        nok_dims = False
        if self.iref_probe>self.dims[0]:
            nok_dims = True
        
        if nok_dims:
            print('Mesh dims are ({0:d},{1:d},{2:d})'.format(
                    self.dims[0],self.dims[1],self.dims[2]))
            print('reference probe is in i_ref = {0:d}'.format(self.iref_probe))
            message = 'Reference probe is not on the defined grid. Please check!'
            raise ValueError(message)
        
    def compute_disp(self):
        if self.coords is None:
            self.read_coordinates()
        
        nx_dn = self.dims[0] - self.iref_probe - 2
        nx_up = self.iref_probe - 2
        nbi = min(nx_dn,nx_up)
        
        nbk = self.dims[2]//2
        print('Selected points in streamwise: +/- {0:d}'.format(nbi))
        print('Selected points in spanwise: +/- {0:d}'.format(nbk))
        
        self.disps = dict()
        self.disps['nbi']=[self.iref_probe-nbi,self.iref_probe+nbi+1]
        self.disps['nbk']=nbk
        self.disps['dx'] = 0.5 * (self.coords['x'][
                                    (self.disps['nbi'][0]+1):(self.disps['nbi'][1]+1),:-1,1:-1] 
                                - self.coords['x'][
                                    (self.disps['nbi'][0]-1):(self.disps['nbi'][1]-1),:-1,1:-1]) 
        self.disps['dy'] = ( self.coords['y'][self.disps['nbi'][0]:self.disps['nbi'][1],1:,1:-1]
                            - self.coords['y'][self.disps['nbi'][0]:self.disps['nbi'][1],:-1,1:-1] )
        self.disps['dz'] = 0.5 * (self.coords['z'][self.disps['nbi'][0]:self.disps['nbi'][1],:-1,2:] 
                                - self.coords['z'][self.disps['nbi'][0]:self.disps['nbi'][1],:-1,:-2]) 
        nn = (self.disps['dx']**2 + self.disps['dy']**2)**0.5
        self.disps['nx'] = self.disps['dx'] / nn
        self.disps['ny'] = self.disps['dy'] / nn
        self.disps['xslice'] = slice(self.disps['nbi'][0],self.disps['nbi'][1])
        self.disps['zslice'] = slice(1,self.dims[2]-1)
        self.disps['iref'] = self.disps['nx'].shape[0]//2
        self.disps['kref'] = self.disps['nx'].shape[2]//2
    
    def get_ut_un(self,layer):
        from numpy import newaxis
        if self.disps is None:
            self.compute_disp()
        if self.fid is None:
            self.open_file()
        if layer < self.dims[1]:
            layer_name = 'layer_{0:06d}'.format(layer)
        tmp = self.fid['{0:s}/x_velocity'.format(layer_name)][()]
        ux = tmp[:,self.disps['xslice'],self.disps['zslice']]
        tmp = self.fid['{0:s}/y_velocity'.format(layer_name)][()]
        uy = tmp[:,self.disps['xslice'],self.disps['zslice']]
        ut = - ux * self.disps['ny'][newaxis,:,0,:] + uy * self.disps['nx'][newaxis,:,0,:]
        un = ux * self.disps['nx'][newaxis,:,0,:] + uy * self.disps['ny'][newaxis,:,0,:]
        ut_m = ut.mean(axis=0)
        un_m = un.mean(axis=0)
        ut -= ut_m[newaxis,:,:]
        un -= un_m[newaxis,:,:]
        return ut,un
    
    def compute_R11_R22(self,layer):
        from numpy import corrcoef,zeros,diagonal
        from copy import deepcopy
        iref = self.disps['iref']
        kref = self.disps['kref']
        R11 = zeros((self.disps['nx'].shape[0],self.disps['nx'].shape[2]))
        R22 = zeros((self.disps['nx'].shape[0],self.disps['nx'].shape[2]))
        nz = self.disps['nx'].shape[2]
        ut,un = self.get_ut_un(layer)
        for idx in range(self.disps['nx'].shape[0]):
            tmp = corrcoef(ut[:,iref,kref],ut[:,idx,:],rowvar=False)
            R11[idx,:] = deepcopy(tmp[1:,0])
            tmp = corrcoef(un[:,iref,kref],un[:,idx,:],rowvar=False)
            R22[idx,:] = deepcopy(tmp[1:,0])
        return R11,R22
        
    def get_ref_layer_radius(self,layer):
        from numpy import array,cross,einsum
        from numpy.linalg import norm,inv
        iref = self.disps['iref']
        kref = self.disps['kref']
        
        xx = self.coords['x'][self.disps['xslice'],layer,self.disps['zslice']]
        yy = self.coords['y'][self.disps['xslice'],layer,self.disps['zslice']]
        zz = self.coords['z'][self.disps['xslice'],layer,self.disps['zslice']]

        xvec = array((xx,yy,zz)) # shape (3,nx,nz)
        x1 = 0.5*(xvec[:,iref+1,kref]-xvec[:,iref-1,kref])
        x1 /= norm(x1)
        z1 = 0.5*(xvec[:,iref,kref+1]-xvec[:,iref,kref-1])
        z1 /= norm(z1)
        y1 = cross(z1,x1)

        B1 = array((x1,y1,z1))

        Xvec = einsum('ij,jkl->ikl',B1,xvec)

        r1 = Xvec[0,:,:] - Xvec[0,iref,kref]
        r2 = Xvec[1,:,:] - Xvec[1,iref,kref]
        r3 = Xvec[2,:,:] - Xvec[2,iref,kref]
        return r1,r2,r3
