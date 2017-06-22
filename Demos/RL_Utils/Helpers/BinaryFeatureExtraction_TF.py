import tensorflow as tf

'''
# TODO Rewrite specification
# TODO Add tf scope
'''

class BinaryFeatureExtractor:
    
    def __init__(self,GD=None):
        
        # Subspace membership list
        self.M = None
        self.M_tf = None
        
        # Magnitude of raw feature space
        if GD != None:
            assert len(GD) == 2, "Error, initializer requires a list of [Mag raw feat space, Depth of subspaces]"
            self.initBinaryFeatureList(GD[0],GD[1])
        else:
            self.G = None
            self.D = None
        
            # Magnitude of binary feature space
            self.T = None
            
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
        # Save magnitude of raw feature space (G)
        self.G = G
        
        # Save depth of subspace hierarchy (D)
        self.D = D
        
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
        
        # Calculate and save magnitue of binary feature space (T)
        self.T = sum(C)
        
        # Generate deactivate binary feature set
        self.A = [False for i in range(self.T)]
        
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
        assert len(self.M) == self.T
        
        for i in range(self.T):
            if (v >= self.M[i][0]) and (v <= self.M[i][1]):
                self.A[i] = True
            else:
                self.A[i] = False
        
        return self.A
                
    def activateBinaryFeatures_TF(self, v):
        '''
        
        '''
        with tf.name_scope("activate_binary_features"):
            # Slice left column of range mins
            lcol = tf.slice(self.M_tf, [0,0], [tf.shape(self.M_tf)[0], 1])
            # Slice right column of range maxs
            rcol = tf.slice(self.M_tf, [0,1], [tf.shape(self.M_tf)[0], 1])
            # Check if val >= min
            lval = tf.greater_equal(v, tf.cast(lcol,tf.int64))
            # Check if val <= max
            rval = tf.less_equal(v, tf.cast(rcol,tf.int64))  
            # Ensure both rval and lval are true
            features = tf.logical_and(lval, rval)
        
        return features
    
    def getBinaryFeatureSpaceSize(self):
        assert self.T != None
        return self.T
    
    def getBinaryFeatureFullMatrix_TF(self):
        assert self.G != None
        return tf.constant([list(self.activateBinaryFeaturesBrute(i)) for i in range(self.G)])
    
    
    ##JCR
    def map_features_binary(state,action_,user_fbin_baseline,Fs=250,L=1000):
        #generate the number of fft points per freq bin
        pp_fbin = generate_frequency_bins(L,Fs,fbin_min,fbin_max,fbin_steps)
        
        #generate a segment mapping for the tensor - like pp_fbin but each element in this array corresponds to a single element of the tensor
        segmap = generate_segment_map(pp_fbin,1)
    
        with tf.name_scope("Map_Binary_Features"):
    
            #generate fft
            spectro=tf.fft(tf.slice(state,[0,0],[-1, L]),name="raw_fft")
    
            #transpose the sensor so that we can use segment_sum below
            spectro=tf.transpose(spectro,name="pre_binning_transpose")
            
            #reduce the tensor using binning
            spectro_binned = tf.segment_sum(spectro,segmap,name="f_binning")
        
            #remove the first and last two bins, corresponding to <fbin_min, >fbin_max, and other fft half
            spectro_binned = spectro_binned[1:-2]
            
            #make the tensor 
            spectro_binned = tf.transpose(spectro_binned,name="post_binning_transpose")
    
            #take fft power, transfer to new variable (no longer processing fft)
            state_features_tmp = tf.abs(spectro_binned)
            
            with tf.name_scope("Generate_Binary_State_Features"):
            
                #Extend this out so we can do a simple max with user baseline bins
                state_features_tmp = tf.tile(state_features_tmp,[1,feat_per_fbin_per_ch],name="expand_state_feat")
        
                #flatten the state features
                #NOTE: state_features_tmp should be #elec x #feat_per_fbin x #fbins
                state_features_tmp = tf.reshape(state_features_tmp,[-1,feat_per_fbin_per_ch,fbin_steps],name="mfb_reshape_state_feat")
            
                #TODO ensure theat GTE is right order on arguments
                GTE=tf.greater_equal(user_fbin_baseline,state_features_tmp)
                GTE=tf.cast(GTE,dtype=tf.int8)
                
                GTE_shifted=tf.pad(GTE, [[0,0],[1,0],[0,0]], mode='CONSTANT')
                GTE_shifted=tf.slice(GTE_shifted,[0,0,0],GTE.get_shape())
                state_features_bin=tf.not_equal(GTE,GTE_shifted,"isolate_tf")
                state_features_bin = tf.reshape(state_features_bin,[-1],name='flatten_state_space_tensor')
            
            with tf.name_scope("Generate_Binary_Action_Features"):
                action_features_bin = act_to_actbin(action_)
                action_features_bin = tf.reshape(action_features_bin,[-1],name='flatten_act_space_tensor')    
        
            features = tf.concat([state_features_bin,action_features_bin],0)
            return features

def build_graph_binary_feature_extract():
    global z
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
        bfe = BinaryFeatureExtractor([21, 1])
        #bfe.initBinaryFeatureList(21, 3)
        bfe.activateBinaryFeaturesBrute(10)
        g = bfe.A
        print(g)
        
        weights=tf.constant([[1 for i in range(21)]])
        b=bfe.getBinaryFeatureFullMatrix_TF()        
        b=tf.cast(b,tf.int32)
        c = tf.matmul(weights,b,False,True)
        d = bfe.activateBinaryFeatures_TF(tf.constant(11, dtype=tf.int64))

        
        init=tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)    
        z=sess.run([b,c,d])
        sess.close()
    
#build_graph_binary_feature_extract() 




