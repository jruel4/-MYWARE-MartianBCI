import tensorflow as tf

'''
# TODO Rewrite specification
# TODO Add tf scope
'''

class BinaryFeatureExtractor:
    
    def __init__(self):
        
        # Subspace membership list
        self.M = None
        self.M_tf = None
        
        # Magnitude of binary feature space
        self.G = None
        
        # Activated binary features
        self.A = None

    def initBinaryFeatureList(self, G, D):
        '''
        Purpose: Calculates subspaces given input assumptions
                 and returns a list of binary feature membership lists, one for each subspace.
        Inputs: 
            G <= Magnitude of raw feature space
            D <= Depth of subspace hierarchy, i.e. # of levels
            
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
                subspace_members = [b, b+size-1]
                M += [subspace_members]
        
        self.M = M
        
        # Calculate and save magnitue of binary feature space (G)
        self.G = sum(C)
        
        # Generate deactivate binary feature set
        self.A = [False for i in range(self.G)]
        
        # Generate tensor for M
        self.M_tf = tf.constant(self.M)

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
            if (v >= self.M[i][0]) and (v <= self.M[i][1]):
                self.A[i] = [True]
            else:
                self.A[i] = [False]
        
        return self.A
                
    def activateBinaryFeatures_TF(self, v):
        '''
        
        '''
        # Slice left column of range mins
        lcol = tf.slice(self.M_tf, [0,0], [tf.shape(self.M_tf)[0], 1])
        # Slice right column of range maxs
        rcol = tf.slice(self.M_tf, [0,1], [tf.shape(self.M_tf)[0], 1])
        # Check if val >= min
        lval = tf.greater_equal(tf.constant(v), lcol)
        # Check if val <= max
        rval = tf.less_equal(tf.constant(v), rcol)  
        # Ensure both rval and lval are true
        features = tf.logical_and(lval, rval)
        
        return features
            

bfe = BinaryFeatureExtractor()
bfe.initBinaryFeatureList(21, 3)
bfe.activateBinaryFeaturesBrute(10)
g = bfe.A







