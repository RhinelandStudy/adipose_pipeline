
# Copyright 2023 Population Health Sciences and AI in Medical Imaging, German Center for Neurodegenerative Diseases (DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:21:42 2017

@author: shahidm
"""

from Crypto.Cipher import AES
import base64
import os,random,string

try:
    from configparser import SafeConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser
    

class Configurator():
    def __init__(self, config_file=None):
        
        self.parser = SafeConfigParser()
        config_files= [os.path.join(os.path.expanduser("~"), ".rsutils.cnf"), os.path.join(os.getcwd(),".rsutils.cnf")]
        
        if config_file:
            config_files.insert(0, os.path.abspath(config_file))
	#print "Configurator: config_files are:",config_files
        self.parser.read(config_files)
        
        self.BLOCK_SIZE = 16
        pad = lambda s: s + (self.BLOCK_SIZE - len(s) % self.BLOCK_SIZE) * '{'
        self.EncodeAES = lambda c, s: base64.b64encode(c.encrypt(pad(s)))
        self.DecodeAES = lambda c, e: c.decrypt(base64.b64decode(e)).rstrip('{')
        #self.secret = '0123456789abcdef' #os.urandom(BLOCK_SIZE)#
        self.rkey = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
        #self.cipher = AES.new(self.secret)
        
    def get_configuration(self, section):

        options = {}
        
        if self.parser.has_section(section):
            for name,val in self.parser.items(section):
                if name == 'xnat_password':
                    self.parser.set(section, 'key',self.rkey)
                    options['key'] = self.rkey
                    options[name] = self.mask_string(self.rkey,val)
                else:
                    options[name]=val
        else:
            raise ValueError("No confguration options found for section %s" %section)
            
        return options
        
    def mask_string(self, key, string, encdec='enc'):
        cipher = AES.new(key)
        if encdec == 'enc':
            return self.EncodeAES(cipher, string)
        else:
            return self.DecodeAES(cipher, string)
        
        
