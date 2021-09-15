from distutils.core import setup
  

setup(name='nn_service',
      version='1.0.0',
      install_requires=[
            'torch',
            'wandb',
            'pytorch_lightning',
            'torch-geometric',
            'pyarrow',
            'simple_zmq'
      ],
      description='Simple neural net server-client implementation',
      author='Jacky Liang',
      author_email='jackyliang@cmu.edu',
      url='none',
      packages=['nn_service']
     )

