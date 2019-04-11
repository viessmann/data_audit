from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='data_audits',
      version='0.1.0',
      description='Viessmann Data Audits',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
      ],
      keywords='data science',
      author='GhlT',
      author_email='ghlt@viessmann.com',
      license='Other/Proprietary License',
      packages=find_packages(),
      install_requires=[
          'pandas>=0.22',
          'numpy>=1.14',
          'scikit-learn>=0.19',
          'seaborn>=0.8.1'
      ],
      include_package_data=True,
      zip_safe=False)