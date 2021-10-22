#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:12:47 2021

@author: greydon
"""

import os
import pathlib
from typing import List, Union
import pandas as pd
import nibabel as nib
import pydicom as dcm
import subprocess



def _prepare_subject(subj_dir: Union[str, PathLike],
					 timeout: Union[str, PathLike] = None,
					 confirm_uids: bool = True) -> str:
	"""
	Reorganizes `subj_dir` to structure more compatible with ``heudiconv``

	Parameters
	----------
	subj_dir : str or pathlib.Path
		Path to subject directory as downloaded from ppmi-info.org
	timeout : str or pathlib.Path, optional
		If set, path to directory where bad scans should be moved. Default:
		None
	confirm_uids : bool, optional
		Whether to check that DICOM study instance UIDs for provided subject
		are all consistent for a given session. Only applicable if `pydicom`
		is installed. Default: True

	Returns
	-------
	subject : str
		Basename of `subj_dir`
	force : list
		List of paths to data directories where sessions had inconsistent
		study instance UIDs
	"""

	bad_scans = pd.read_csv(BAD_SCANS)

	# coerce subj_dir to path object
	subj_dir = pathlib.Path(subj_dir).resolve()

	# get all scan types for subject
	scans = [f for f in subj_dir.glob('*') if not f.name.isdigit()]

	# if subject was previously converted update number structure correctly
	# FIXME: should we check to see if there's overlap (i.e., a new scan was
	# added from the same session?); could pull study UID / date from dicoms?
	prev = len([f for f in subj_dir.glob('*') if f.name.isdigit()])

	# get all sessions for subject (session = same day)
	sessions = [v.name[:7] for v in subj_dir.rglob('????-??-??_??_??_??.?')]
	sessions = sorted(set(sessions))

	# iterate through sessions and copy scans to uniform directory structure
	force = []
	for n, ses in enumerate(sessions, prev + 1):
		sids = set()

		# make session directory
		ses_dir = subj_dir / str(n)
		ses_dir.mkdir(exist_ok=True)

		# iterate through all scans for a given session (visit) and move
		for scan_type in subj_dir.glob('*/{0}*/*'.format(ses)):
			# idk why this would be but check just in case????
			if not scan_type.is_dir():
				continue

			# if this is a bad scan, move it to `timeout`
			if scan_type.name in bad_scans.scan.values and timeout is not None:
				dest = timeout / subj_dir.name / ses_dir.name
				dest.mkdir(parents=True, exist_ok=True)
				scan_type.rename(dest / scan_type.name)
			# otherwise, move it to the appropriate scan directory
			else:
				if confirm_uids:
					for img in scan_type.glob('*dcm'):
						img = dcm.read_file(str(img), stop_before_pixels=True)
						sids.add(img[('0020', '000d')].value)
				out = ses_dir / scan_type.name
				scan_type.rename(out)

			# if there are no more scans in the parent directory, remove it
			remain = [f for f in scan_type.parent.glob('*') if f != scan_type]
			if len(remain) == 0:
				scan_type.parent.rmdir()

		if len(sids) > 1:
			force.append(out.parent)

	# remove empty directories
	for scan in scans:
		scan.rmdir()

	return subj_dir.name, force


def _prepare_directory(data_dir: Union[str, PathLike],
					   ignore_bad: bool = True,
					   confirm_uids: bool = True) -> List[str]:
	"""
	Reorganizes PPMI `data_dir` to a structure compatible with ``heudiconv``

	PPMI data starts off with a sub-directory structure that is not conducive
	to use with ``heudiconv``. By default, scans are grouped by scan type
	rather than by session, and there are a number of redundant sub-directories
	that we don't need. This script reorganizes the data, moving things around
	so that the general hierarchy is {subject}/{session}/{scan}, which makes
	for a much easier time converting the PPMI dataset into BIDS format.

	An added complication is that a minority of the scans in the PPMI database
	are "bad" to some degree. For most, it is likely that there was some issue
	with exporting/uploading the DICOM files. For others, the conversion
	process we intend to utilize (``heudiconv`` and ``dcm2niix``) fails to
	appropriately convert the files due to some idiosyncratic reason that could
	be fixed but we don't have the patience to fix at the current juncture.
	Nonetheless, these scans need to be removed so that we can run the batch of
	subjects through ``heudiconv`` without any abrupt failures. By default,
	these scans are moved to a sub-directory of `data_dir`; setting
	`ignore_bad` to False will retain these scans (but be warned!)

	Parameters
	----------
	data_dir : str or pathlib.Path
		Filepath to PPMI dataset, as downloaded from https://ppmi-info.org
	ignore_bad : bool, optional
		Whether to ignore "bad" scans (i.e., ones that are known to fail
		conversion or reconstruction)
	confirm_uids : bool, optional
		Whether to check that DICOM study instance UIDs for provided subject
		are all consistent for a given session. Only applicable if `pydicom`
		is installed. Default: True

	Returns
	-------
	subjects : list
		List of subjects who are ready to be converted / reconstructed with
		``heudiconv``
	coerce : list
		List of paths to data directories where subjects / sessions may have
		had inconsistent study instance UIDs that should be coerced
	"""

	if isinstance(data_dir, str):
		data_dir = pathlib.Path(data_dir).resolve()

	# location where "bad" scans will be moved
	if ignore_bad:
		timeout = data_dir / 'bad'
		timeout.mkdir(exist_ok=True)
	else:
		timeout = None

	subjects, coerce = [], []
	for subj_dir in sorted(data_dir.glob('*')):
		if not subj_dir.is_dir() or subj_dir.name == 'bad':
			continue
		subj, force = _prepare_subject(subj_dir, timeout=timeout,
									   confirm_uids=confirm_uids)
		subjects.append(subj)
		coerce.extend(force)

	return subjects, coerce


def _force_consistent_uids(data_dir: Union[str, PathLike],
						   target_uid: str = None):
	"""
	Forces all DICOMs inside `data_dir` to have consistent study instance UID

	Will recursively crawl sub-directories of `data_dir` to check for DICOMs

	Parameters
	----------
	data_dir : str or pathlib.Path
		Path to data directory containg DICOMs that should be overwritten with
		new study instance UID.
	target_uid : str, optional
		Study instance UID to assign to all DICOMs found in `data_dir`. If not
		specified the study instance UID from the first DICOM found will be
		used instead. Default: None
	"""

	data_dir = pathlib.Path(data_dir).resolve()

	for n, fn in enumerate(data_dir.rglob('*dcm')):
		img = dcm.read_file(str(fn))
		if target_uid is None and n == 0:
			target_uid = str(img[('0020', '000d')].value)
			continue
		img[('0020', '000d')].value = target_uid
		dcm.write_file(str(fn), img)


def _clean_directory(out_dir: Union[str, PathLike]):
	"""
	Does some final post-processing on the converted data

	Includes merging T1w images that were, for some inexplicable reason, split
	into two volumes

	Parameters
	----------
	out_dir : str or pathlib.Path
		Path to output directory where BIDS-format PPMI dataset should be
		generated
	"""

	return out_dir


def _merge_3d_t1w(filename: Union[str, PathLike]) -> pathlib.Path:
	"""
	Merges T1w images that have been split into two volumes

	Parameters
	----------
	filename : str or pathlib.Path
		Path to T1w image that needs to be merged

	Returns
	-------
	filename : pathlib.Path
		Path to merged T1w image
	"""

	import numpy as np

	filename = pathlib.Path(filename).resolve()
	img = nib.load(str(filename))
	if not (len(img.shape) == 4 and img.shape[-1] > 1):
		return

	# split data along fourth dimension and then concatenate along third
	imdata = img.get_data()
	cat = [d.squeeze() for d in np.split(imdata, imdata.shape[-1], axis=-1)]
	imdata = np.concatenate(cat, axis=-1)

	new_img = img.__class__(imdata, img.affine, img.header)
	nib.save(new_img, filename)

	return filename

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    rc = process.poll()
    stdout = process.communicate()[0]
    return rc

#%%



missing = 'B0rf Map'

subjects_remove=['100889','100878','101124','101175']

dataset_path = r'/home/greydon/Documents/datasets/PPMI'


BAD_SCANS = os.path.join(dataset_path, 'sessions.txt')
HEURISTIC = os.path.join(dataset_path, 'clinical_imaging.py')
raw_dir = pathlib.Path(os.path.join(dataset_path, 'sourcedata')).resolve()
out_dir = pathlib.Path(os.path.join(dataset_path, 'bids')).resolve()
coerce_study_uids = False
overwrite = False


# generate this, if it doesn't already exist
out_dir.mkdir(exist_ok=True)

subjects_, coerce = _prepare_directory(raw_dir, ignore_bad=True,confirm_uids=coerce_study_uids)
subjects = [x for x in subjects if x not in subjects_remove]
# force consistent study UID if desired
if coerce_study_uids:
	for path in coerce:
		_force_consistent_uids(path)


# run heudiconv over all potential sessions
for session in range(1, 6):
	
	command=' '.join([
		'heudiconv',
		'-d', f'"{dataset_path}'+'/sourcedata/{subject}/{session}/*/*dcm"',
		f"-s {' '.join(subjects[11:30])}",
		'-ss', str(session),
		'-o', out_dir.as_posix(),
		'-f', HEURISTIC,
		'-c', 'dcm2niix',
		'-b'
	])
	
	run_command(command)
	

out_dir = _clean_directory(out_dir)

