from distutils.core import setup
setup(
  name = 'torch_kit',
  packages = ['torch_kit'],
  version = '0.0.0.1',
  description = 'Torch Kit: PyTorch Utilities',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/torch_kit',
  download_url = 'https://github.com/brookisme/torch_kit/tarball/0.1',
  keywords = ['PyTorch','CNN','Neural Networks','Machine learning','Deep learning'],
  include_package_data=True,
  data_files=[
    (
      'config',[]
    )
  ],
  classifiers = [],
  entry_points={
      'console_scripts': [
        'torch_kit=torch_kit.cli.cli:cli'
      ]
  }
)