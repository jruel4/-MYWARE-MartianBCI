import tensorflow as tf

class BinaryFeatureExtractor:
    
    def __init__(self):
        
        # Subspace membership list
        self.M = None
        
        # Magnitude of binary feature space
        self.G = None
        
        # Activated binary features
        self.A = None

    def initBinaryFeatureList(self, G, D, O):
        '''
        Purpose: Calculates subspaces given input assumptions
                 and returns a list of binary feature membership lists, one for each subspace.
        Inputs: 
            G <= Magnitude of raw feature space
            D <= Depth of subspace hierarchy, i.e. # of levels
            O <= Difference amount b/w adjacent subspaces
            
        Outputs:
            M <= subspace membership list, one for each subspace.
        '''
        # Calculate hierarchy depth stride (H)
        H = int(G/D)
        
        # Calculates size of the subspaces in each of the (D) subspace hierarchy levels
        N = [(i-1)*H+1 for i in range(1,D+1)]
        
        # Calculate number of subspaces (C) in each of the (D) subspace hierarchy levels 
        C = [G-Ni+1 for Ni in N]
        
        # Calculate the membership set (M) for each subspace
        M = list()
        for level in range(D):
            size = N[level]
            for b in range(C[level]):
                subspace_members = list(range(b, b+size))
                M += [subspace_members]
        
        self.M = M
        
        # Calculate and save magnitue of binary feature space (G)
        self.G = sum(C)
        
        # Generate deactivate binary feature set
        self.A = [False for i in range(self.G)]

    def activateBinaryFeaturesBrute(self, v):
        '''
        Purpose: map an actual feature space value (v) to its corresponding
                 binary features. *Use brute force for membership checking*
        
        Inputs: v <= actual feature space value
        
        Outputs: activated_feature_list <= list of binary features with those
                 corresponding to (v) set as True
        '''
        assert self.M != None
        assert len(self.M) == len(self.A)
        assert len(self.M) == self.G
        
        for i in range(self.G):
            if v in self.M[i]:
                self.A[i] = True
            else:
                self.A[i] = False
                
            
    
bfe = BinaryFeatureExtractor()
bfe.initBinaryFeatureList(21, 3, 1)
bfe.activateBinaryFeaturesBrute(10)








