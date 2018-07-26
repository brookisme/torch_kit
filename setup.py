from distutils.core import setup
setup(
  name = 'pytorch_nns',
  packages = ['pytorch_nns'],
  version = '0.0.0.1',
  description = 'PyTorch Neural Networks: models and utilities',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/pytorch-nns',
  download_url = 'https://github.com/brookisme/pytorch-nns/tarball/0.1',
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
      ]
  }
)