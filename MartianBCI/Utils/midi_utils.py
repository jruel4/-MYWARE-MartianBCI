# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:44:52 2017

@author: marzipan
"""

import rtmidi

def send_midi_msg(ch=0x90, note=60, vel=112):
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    print(available_ports)
    port_no = input("Enter desired port number: ")
    midiout.open_port(port_no)
    note_on = [ch, note, vel] # channel 1, middle C, velocity 112
    midiout.send_message(note_on)
    #del midiout # Deleting this creates a bunch of messages that I dont' yet uynderstand
    
    