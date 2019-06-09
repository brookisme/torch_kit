from distutils.core import setup
setup(
  name = 'antorchita',
  packages = ['antorchita'],
  version = '0.0.0.1',
  description = 'PyTorch Neural Networks: models and utilities',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/antorchita',
  download_url = 'https://github.com/brookisme/antorchita/tarball/0.1',
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