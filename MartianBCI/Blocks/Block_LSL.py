# -*- coding: utf-8 -*-

# Define imports
import pylsl
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet


if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
    

class Block_LSL(Block):
    def __init__(self, _pipe, _parent_UID, _parent_output_key, stream_name, stream_type='EEG',stream_fs=250):

        self.mPipeline = _pipe
        self.mParentOutputKey = _parent_output_key
        self.mParentUID = _parent_UID
        
        #Verify parent block output is valid for key we were given
        parent_block = self.mPipeline.mBlocks[self.mParentUID]
        parent_output_keys = parent_block['func'].get_output_struct().keys()

        if self.mParentOutputKey not in parent_output_keys:
            raise KeyError("Block_LSL: parentOutputKey:" + str(self.mParentOutputKey) + " is invalid, valid keys are: ", parent_output_keys)

        self.mParentOutputLen = parent_block['func'].get_output_struct()[self.mParentOutputKey]
        
        if not isinstance(self.mParentOutputLen, int):
            raise TypeError("Block_LSL: Output length must be an int, type is: ", type(self.mParentOutputLen), " value is: ", self.mParentOutputLen)
        
        if not isinstance(stream_name, str) or not isinstance(stream_type,str):
            raise UserWarning("Block_LSL: Stream type/name is not a string, name: " + str(stream_name) + " type:" + str(stream_type))
        
        
        print(str(stream_name), str(stream_type), self.mParentOutputLen, stream_fs, 'float32', str(stream_name) + time.strftime("_%d.%m.%Y-%H:%M:%S"))
        self.mInfo = StreamInfo(str(stream_name), str(stream_type), self.mParentOutputLen, stream_fs, 'float32', str(stream_name) + time.strftime("_%d.%m.%Y-%H:%M:%S"))
        self.mOutlet = StreamOutlet(self.mInfo)
        return
    
    def run(self, buf):
        assert isinstance(self.mOutlet, pylsl.StreamOutlet)
        
        try:
            self.mOutlet.push_sample(buf[self.mParentOutputKey])
        except ValueError as e:
            print(e)
            print("Received data.shape: ", np.shape(buf[self.mParentOutputKey]))

    def get_output_struct(self):
        return {}