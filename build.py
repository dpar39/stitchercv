#!/usr/bin/env python
"""
Builds this project from the ground up
"""
import os
import re
import sys
import glob
import json
import base64
import shutil
import zipfile
import tarfile
import argparse
import threading
import subprocess
import multiprocessing

try:  # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:   # Fall back to Python 2's urllib2
    from urllib2 import urlopen

# Configuration
OPENCV_SRC_URL = 'https://github.com/opencv/opencv/archive/4.0.1.zip'
OPENCVCONTRIB_SRC_URL = 'https://github.com/opencv/opencv_contrib/archive/4.0.1.zip'
GMOCK_SRC_URL = 'https://github.com/google/googletest/archive/release-1.8.1.zip'

IS_WINDOWS = sys.platform == 'win32'
if sys.platform == 'win32':
    PLATFORM = 'windows'
elif 'linux' in sys.platform:
    PLATFORM = 'linux'
elif sys.platform == 'darwin':
    PLATFORM = 'darwin'


def which(program):
    """
    Returns the full path of to a program if available in the system PATH, None otherwise
    """
    def is_exe(fpath):
        """
        Returns true if the file can be executed, false otherwise
        """
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def link_file(src_file_path, dst_link):
    if not os.path.exists(src_file_path):
        raise FileNotFoundError(src_file_path)
    print('Creating link for file "%s" in "%s"' % (src_file_path, dst_link))
    if IS_WINDOWS:
        shutil.copyfile(src_file_path, dst_link)
        return
        link_cmd = 'mklink "%s" "%s"' % (dst_link, src_file_path)
    else:
        link_cmd = 'ln -sf "%s" "%s"' % (src_file_path, dst_link)
    os.system(link_cmd)


class ShellRunner(object):
    def __init__(self, arch_name):
        self._env = os.environ.copy()
        self._extra_paths = []
        self._arch_name = arch_name
        if IS_WINDOWS:
            self._detect_vs_version()
        # Add tools like ninja and swig to the current PATH
        this_dir = os.path.dirname(os.path.realpath(__file__))
        tools_dir = os.path.join(this_dir, 'thirdparty', 'tools', PLATFORM)
        self.add_system_path(tools_dir)

    def add_system_path(self, new_path, at_end=True):
        curr_path_str = self._env['PATH']
        path_elmts = set(curr_path_str.split(os.pathsep))
        if new_path in path_elmts:
            return
        if at_end:
            self._env['PATH'] = curr_path_str + os.pathsep + new_path
        else:
            self._env['PATH'] = new_path + os.pathsep + curr_path_str

    def set_env_var(self, var_name, var_value):
        assert isinstance(var_name, str), 'var_name should be a string'
        assert isinstance(
            var_value, str) or var_value is None, 'var_value should be a string or None'
        self._env[var_name] = var_value

    def get_env_var(self, var_name):
        return self._env.get(var_name, '')

    def get_env(self):
        return self._env

    def run_cmd(self, cmd_args, cmd_print=True, cwd=None, input=None):
        """
        Runs a shell command
        """
        if isinstance(cmd_args, str):
            cmd_args = cmd_args.split()
        cmd_all = []
        if IS_WINDOWS:
            cmd_all = [self._vcvarsbat, self._arch_name,
                       '&&', 'set', 'CL=/MP', '&&']
        cmd_all = cmd_all + cmd_args

        if cmd_print:
            print(' '.join(cmd_args))

        p = subprocess.Popen(cmd_all, env=self._env, cwd=cwd, shell=True,
                             stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        if input:
            p.communicate(input=input)
        else:
            p.wait()
        if p.returncode != 0:
            print('Command "%s" exited with code %d' %
                  (' '.join(cmd_args), p.returncode))
            sys.exit(p.returncode)

    def _detect_vs_version(self):
        """
        Detects the first available version of Visual Studio
        """
        vc_releases = [
            ('Visual Studio 15 2017',
             r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvarsall.bat'),
            ('Visual Studio 15 2017',
             r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat'),
            ('Visual Studio 14 2015', r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat')]
        for (vsgenerator, vcvarsbat) in vc_releases:
            if os.path.exists(vcvarsbat):
                self._vcvarsbat = vcvarsbat
                self._vc_cmake_gen = vsgenerator
                if "64" in self._arch_name:
                    self._vc_cmake_gen += ' Win64'
                break

    def get_vc_cmake_generator(self):
        return self._vc_cmake_gen


class Builder(object):
    """
    Class that holds the whole building process
    """

    def repo_path(self, rel_path=''):
        if not rel_path:
            return self._root_dir
        return os.path.join(self._root_dir, rel_path).replace('\\', '/')

    def build_name(self):
        return PLATFORM + '_' + self._build_config + '_' + self._arch_name

    def build_dir_name(self, prefix):
        """
        Returns a name for a build directory based on the build configuration
        """
        return os.path.join(prefix, 'build_' + self.build_name())

    def build_path(self, rel_path):
        return os.path.join(self._build_dir, rel_path).replace('\\', '/')

    def run_cmake(self, cmake_generator='Ninja', cmakelists_path='.'):
        """
        Runs CMake with the specified generator in the specified path with
        possibly some extra definitions
        """
        cmake_args = ['cmake',
                      '-DCMAKE_INSTALL_PREFIX=' + self._install_dir,
                      '-DCMAKE_PREFIX_PATH=' + self._install_dir,
                      '-DCMAKE_BUILD_TYPE=' + self._build_config,
                      '-G', cmake_generator, cmakelists_path]
        self.run_cmd(cmake_args)

    def run_cmd(self, cmd_args, cmd_print=True, cwd=None, input=None):
        self._shell.run_cmd(cmd_args, cmd_print=cmd_print,
                            cwd=cwd, input=input)

    def set_startup_vs_prj(self, project_name):
        """
        Rearranges the projects so that the specified project is the first
        therefore is the startup project within Visual Studio
        """
        solution_file = glob.glob(self._build_dir + '/*.sln')[0]
        sln_lines = []
        with open(solution_file) as file_handle:
            sln_lines = file_handle.read().splitlines()
        lnum = 0
        lin_prj_beg = 0
        lin_prj_end = 0
        for line in sln_lines:
            if project_name in line:
                lin_prj_beg = lnum
            if lin_prj_beg > 0 and line.endswith('EndProject'):
                lin_prj_end = lnum
                break
            lnum = lnum + 1
        prj_lines = sln_lines[:2] + sln_lines[lin_prj_beg:lin_prj_end + 1] \
            + sln_lines[2:lin_prj_beg] + sln_lines[lin_prj_end + 1:]
        with open(solution_file, "w") as file_handle:
            file_handle.writelines(["%s\n" % item for item in prj_lines])

    def build_googletest(self):
        """
        Extract and build GMock/GTest libraries
        """
        if os.path.isfile(os.path.join(self._third_party_install_dir, 'lib/cmake/GTest/GTestConfig.cmake')):
            return  # We have Gtest installed
        # Download googletest sources if not done yet
        gmock_src_pkg = self.download_third_party_lib(GMOCK_SRC_URL, 'googletest.zip')
        # Get the file prefix for googletest
        gmock_extract_dir = self.get_third_party_lib_dir('googletest')
        if gmock_extract_dir is None:
            # Extract the source files
            self.extract_third_party_lib(gmock_src_pkg)
            gmock_extract_dir = self.get_third_party_lib_dir('googletest')
        # Build GoogleTest/GoogleMock and install
        cmake_extra_defs = [
            '-DCMAKE_INSTALL_PREFIX=' + self._third_party_install_dir,
        ]
        self.build_cmake_lib(gmock_extract_dir, cmake_extra_defs, ['install'])

    def get_third_party_lib_dir(self, prefix):
        """
        Get the directory where a third party library with the specified prefix
        name was extracted, if any
        """
        third_party_dirs = next(os.walk(self._third_party_dir))[1]
        for lib_dir in third_party_dirs:
            if prefix in lib_dir:
                return os.path.join(self._third_party_dir, lib_dir)
        return None

    def build_opencv(self):
        """
        Downloads and builds OpenCV from source
        """
        ocv_build_modules = ['highgui', 'core', 'imgproc', 'stitching', 'imgcodecs']

        # Skip building OpenCV if done already
        if IS_WINDOWS:
            if os.path.exists(os.path.join(self._third_party_install_dir, 'OpenCVConfig.cmake')):
                return
        else:
            lib_files = glob.glob(self._third_party_install_dir + '/lib/libopencv_*.a')
            if len(lib_files) >= len(ocv_build_modules):
                return
        # Download opencv sources if not done yet
        opencv_src_pkg = self.download_third_party_lib(OPENCV_SRC_URL, 'opencv.zip')

        # Download opencv_contrib sources if not done yet
        opencvcontrib_src_pkg = self.download_third_party_lib(OPENCVCONTRIB_SRC_URL, 'opencv_contrib.zip')
        
        # Get the file prefix for OpenCV
        opencv_extract_dir = self.get_third_party_lib_dir('opencv-')

        opencvcontrib_extract_dir = self.get_third_party_lib_dir('opencv_contrib-')

        if opencv_extract_dir is None:
            # Extract the source files
            self.extract_third_party_lib(opencv_src_pkg)
            opencv_extract_dir = self.get_third_party_lib_dir('opencv-')

        if opencvcontrib_extract_dir is None:
            # Extract the source files
            self.extract_third_party_lib(opencvcontrib_src_pkg)
            opencvcontrib_extract_dir = self.get_third_party_lib_dir('opencv_contrib-')

        cmake_extra_defs = [
            '-DCMAKE_INSTALL_PREFIX=' + self._third_party_install_dir,
            '-DBUILD_SHARED_LIBS=OFF',
            '-DBUILD_DOCS=OFF',
            '-DBUILD_PERF_TESTS=OFF',
            '-DWITH_PYTHON=OFF',
            '-DWITH_PYTHON2=OFF',
            '-DWITH_JAVA=OFF',
            '-DBUILD_ZLIB=ON',
            '-DBUILD_ILMIMF=ON',
            '-DBUILD_JASPER=ON',
            '-DBUILD_PNG=ON',
            '-DBUILD_JPEG=ON',
            '-DBUILD_TIFF=ON',
            '-DBUILD_WITH_DEBUG_INFO=OFF',
            '-DBUILD_DOCS=OFF',
            '-DBUILD_TESTS=OFF',
            '-DWITH_FFMPEG=ON',
            '-DWITH_MSMF=OFF',
            '-DWITH_VFW=OFF',
            '-DWITH_OPENEXR=OFF',
            '-DWITH_WEBP=OFF',
            '-DBUILD_opencv_apps=OFF',
            '-DBUILD_opencv_java=OFF',
            '-DBUILD_opencv_python=OFF',
            '-DBUILD_opencv_python2=OFF',
            '-DOPENCV_ENABLE_NONFREE=ON',
            '-DOPENCV_EXTRA_MODULES_PATH=' + opencvcontrib_extract_dir + '/modules'
        ]

        cmake_extra_defs += [
            '-DBUILD_TBB=ON',
            '-DBUILD_LIST=stitching,imgproc,imgcodecs,highgui,videoio,xfeatures2d'
        ]
        if IS_WINDOWS:
            cmake_extra_defs += ['-DBUILD_WITH_STATIC_CRT=ON']

        # Clean and create the build directory
        build_dir = self.build_dir_name(opencv_extract_dir)
        if os.path.exists(build_dir):  # Remove the build directory
            shutil.rmtree(build_dir)
        if not os.path.exists(build_dir):  # Create the build directory
            os.mkdir(build_dir)
        self.build_cmake_lib(opencv_extract_dir, cmake_extra_defs, ['install'], False)

    def get_filename_from_url(self, url):
        """
        Extracts the file name from a given URL
        """
        lib_filename = url.split('/')[-1].split('#')[0].split('?')[0]
        lib_filepath = os.path.join(self._third_party_dir, lib_filename)
        return lib_filepath

    def download_third_party_lib(self, url, package_name=None):
        """
        Download a third party dependency from the internet if is not available offline
        """
        if not package_name:
            lib_filepath = self.get_filename_from_url(url)
        else:
            lib_filepath = os.path.join(self._third_party_dir, package_name)
        if not os.path.exists(lib_filepath):
            print('Downloading %s to "%s" please wait ...' %
                  (url, lib_filepath))
            lib_file = urlopen(url)
            with open(lib_filepath, 'wb') as output:
                output.write(lib_file.read())
        return lib_filepath

    def extract_third_party_lib(self, lib_src_pkg, extract_dir=None):
        """
        Extracts a third party lib package source file into a directory
        """
        if not extract_dir:
            extract_dir = self._third_party_dir
        print('Extracting third party library "%s" into "%s" ... please wait ...' % (
            lib_src_pkg, extract_dir))
        if 'zip' in lib_src_pkg:
            zip_handle = zipfile.ZipFile(lib_src_pkg)
            for item in zip_handle.namelist():
                zip_handle.extract(item, extract_dir)
            zip_handle.close()
        else:  # Assume tar archive (tgz, tar.bz2, tar.gz)
            tar = tarfile.open(lib_src_pkg, 'r')
            for item in tar:
                tar.extract(item, self._third_party_dir)
            tar.close()

    def build_cmake_lib(self, cmakelists_path, extra_definitions, targets, clean_build=False):
        """
        Builds a library using cmake
        """
        build_dir = self.build_dir_name(cmakelists_path)
        # Clean and create the build directory
        # Remove the build directory
        if clean_build and os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        if not os.path.exists(build_dir):  # Create the build directory
            os.mkdir(build_dir)

        # Define CMake generator and make command
        os.chdir(build_dir)
        cmake_cmd = ['cmake', '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=' + self._build_config] + extra_definitions + [cmakelists_path.replace('\\', '/')]

        # Run CMake and Make
        self.run_cmd(cmake_cmd)
        self.run_cmd('ninja')
        for target in targets:
            self.run_cmd(['ninja', target])
        os.chdir(self._root_dir)

    def parse_arguments(self):
        """
        Parses command line arguments
        """
        parser = argparse.ArgumentParser(
            description='Builds the stitching application.')
        parser.add_argument('--arch_name', required=False, choices=['x64', 'x86'],
                            help='Platform architecture', default='x64')
        parser.add_argument('--build_config', required=False, choices=[
                            'debug', 'release'], help='Build configuration type', default='release')
        parser.add_argument('--clean', help='Cleans the whole build directory', action="store_true")
        parser.add_argument('--test', help='Runs unit tests', action="store_true")
        parser.add_argument('--skip_install', help='Skips installation', action="store_true")
        parser.add_argument('--gen_vs_sln', help='Generates Visual Studio solution and projects',
                            action="store_true")

        args = parser.parse_args()

        self._arch_name = args.arch_name
        self._build_clean = args.clean
        self._build_config = args.build_config
        self._gen_vs_sln = args.gen_vs_sln
        self._run_tests = args.test
        self._run_install = not args.skip_install

        # directory suffix for the build and release
        self._root_dir = os.path.dirname(os.path.realpath(__file__))
        self._build_dir = os.path.join(self._root_dir, 'build_' + self.build_name())
        self._install_dir = os.path.join(self._root_dir, 'install_' + self.build_name())
        self._third_party_dir = os.path.join(self._root_dir, 'thirdparty')
        self._third_party_install_dir = os.path.join(
            self._third_party_dir, 'install_' + self.build_name()).replace('\\', '/')

        shell = ShellRunner(self._arch_name)

        # Set up some compiler flags
        if not IS_WINDOWS:
            shell.set_env_var('CXXFLAGS', '-fPIC')
            shell.set_env_var('LD_LIBRARY_PATH', self._install_dir)
        shell.set_env_var('INSTALL_DIR', self._install_dir)
        self._shell = shell

    def build_cpp_code(self):
        """
        Builds the C++ libppp project from sources
        """

        # Build actions
        if self._build_clean and os.path.exists(self._build_dir):
            # Remove the build directory - clean
            shutil.rmtree(self._build_dir)
        if not os.path.exists(self._build_dir):
            # Create the build directory if doesn't exist
            os.mkdir(self._build_dir)

        # Change directory to build directory
        os.chdir(self._build_dir)
        if self._gen_vs_sln:
            # Generating visual studio solution
            cmake_generator = self._shell.get_vc_cmake_generator()
            self.run_cmake(cmake_generator, '..')
            self.set_startup_vs_prj('stitcher_test')
        else:
            targets = ['install']
            cmake_extra_defs = ['-DCMAKE_INSTALL_PREFIX=' + self._install_dir]
            self.build_cmake_lib('..', cmake_extra_defs, targets)
            # Run unit tests for C++ code
            if self._run_tests:
                os.chdir(self._install_dir)
                test_exe = r'.\stitcher_test.exe' if IS_WINDOWS else './ppp_test'
                self.run_cmd([test_exe, '--gtest_output=xml:../tests.xml'])

    def __init__(self):
        # Detect OS version
        self.parse_arguments()

        # Create install directory if it doesn't exist
        if not os.path.exists(self._install_dir):
            os.mkdir(self._install_dir)

        # Build Third Party Libs
        self.build_googletest()
        self.build_opencv()

        # Build this project for a desktop platform (Windows or Unix-based OS)
        self.build_cpp_code()


BUILDER = Builder()
