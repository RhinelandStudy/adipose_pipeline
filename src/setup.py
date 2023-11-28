#!/usr/bin/env python

"""
#Rhineland Study MRI Post-processing pipelines
#rs_adipose_pipeline: Pipeline for segmentation of adipose tissues (FatImaging_F/W) using python/nipype
"""
import os
import sys
from glob import glob
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('adipose_pipeline/models')


def main(**extra_args):
    from setuptools import setup
    setup(name='adipose_pipeline',
          version='1.0.0',
          description='RhinelandStudy Adipose Pipeline',
          long_description="""RhinelandStudy processing for FatImaging scans """ + \
          """It also offers support for performing additional options to run post processing analyses.""" + \
          """More pipelines addition is work in progress.""",
          author= 'estradae',
          author_email='estradae@dzne.de',
          url='http://www.dzne.de/',
          packages = ['adipose_pipeline',
                      'adipose_pipeline.utilities'],
          entry_points={
            'console_scripts': [
                             "run_adipose_pipeline=adipose_pipeline.run_adipose_pipeline:main"
                              ]
                       },
          license='DZNE License',
          classifiers = [c.strip() for c in """\
            Development Status :: 1 - Beta
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Software Development
            """.splitlines() if len(c.split()) > 0],    
          maintainer = 'RheinlandStudy MRI/MRI-IT group, DZNE',
          maintainer_email = 'mohammad.shahid@dzne.de',
          package_data = {'adipose_pipeline': extra_files},
          install_requires=["nipype","nibabel"],
          **extra_args
         )

if __name__ == "__main__":
    main()

