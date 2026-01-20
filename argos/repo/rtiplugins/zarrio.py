# -*- coding: utf-8 -*-

# This file is part of Argos.
#
# Argos is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Argos is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Argos. If not, see <http://www.gnu.org/licenses/>.

"""
    Repository Tree Items (RTIs) for reading Zarr arrays.

    Zarr is a format for the storage of chunked, compressed, N-dimensional arrays.
    See: https://zarr.readthedocs.io/
"""

import logging
import os

import numpy as np

from argos.repo.baserti import BaseRti, shapeToSummary
from argos.repo.iconfactory import RtiIconFactory, ICON_COLOR_UNDEF
from argos.utils.masks import maskedEqual
from argos.utils.defs import DIM_TEMPLATE

logger = logging.getLogger(__name__)

ICON_COLOR_ZARR = '#E377C2'  # Pink/magenta color for Zarr files


def attrsToDict(attrs):
    """ Converts Zarr attributes to a dictionary.

        Reads all attributes and handles errors gracefully.

        :param attrs: Zarr attributes object
        :return: dictionary with attribute key-value pairs
    """
    result = {}
    try:
        for key in attrs.keys():
            try:
                result[key] = attrs[key]
            except Exception as ex:
                logger.warning("Unable to read '{}' attribute: {}".format(key, ex))
                result[key] = str(ex)
    except Exception as ex:
        logger.warning("Unable to iterate attributes: {}".format(ex))
    return result


def zarrArrayUnit(zarrArray):
    """ Returns the unit attribute of a Zarr array.

        Searches for common unit attribute names.

        :param zarrArray: Zarr array object
        :return: unit string or empty string if not found
    """
    attrs = zarrArray.attrs
    for key in ('unit', 'units', 'Unit', 'Units', 'UNIT', 'UNITS'):
        if key in attrs:
            return str(attrs[key])
    return ""


def zarrArrayMissingValue(zarrArray):
    """ Returns the missing data value for a Zarr array.

        Uses fill_value if set, or searches attributes.

        :param zarrArray: Zarr array object
        :return: missing value or None if not defined
    """
    # Check fill_value first (Zarr's native missing value indicator)
    if hasattr(zarrArray, 'fill_value') and zarrArray.fill_value is not None:
        # Zarr uses 0 as default fill_value for numeric types, skip those
        if zarrArray.fill_value != 0:
            return zarrArray.fill_value

    # Search common attribute names
    attrs = zarrArray.attrs
    for key in ('missing_value', 'MissingValue', 'missingValue',
                'FillValue', '_FillValue', 'fill_value'):
        if key in attrs:
            return attrs[key]
    return None


class ZarrArrayRti(BaseRti):
    """ Repository Tree Item (RTI) that contains a Zarr array.
    """
    _defaultIconGlyph = RtiIconFactory.ARRAY

    def __init__(self, zarrArray, nodeName='', fileName='', iconColor=ICON_COLOR_UNDEF):
        """ Constructor.

            :param zarrArray: the underlying Zarr array
            :param nodeName: name of this node in the tree
            :param fileName: path to the Zarr store
            :param iconColor: icon color for this item
        """
        super(ZarrArrayRti, self).__init__(
            nodeName=nodeName,
            fileName=fileName,
            iconColor=iconColor
        )
        self._zarrArray = zarrArray
        self._isStructured = (zarrArray is not None and
                              zarrArray.dtype.names is not None)

    def hasChildren(self):
        """ Returns True if the array has a structured dtype with fields.
        """
        return self._isStructured

    @property
    def isSliceable(self):
        """ Returns True because Zarr arrays can be sliced.
        """
        return True

    def __getitem__(self, index):
        """ Called when using the RTI with an index (e.g., rti[0:10]).

            Applies index to the underlying Zarr array and returns
            the result as a (masked) numpy array.
        """
        array = np.asarray(self._zarrArray[index])
        return maskedEqual(array, self.missingDataValue)

    @property
    def arrayShape(self):
        """ Returns the shape of the underlying array.
        """
        return self._zarrArray.shape

    @property
    def chunking(self):
        """ Returns the chunk sizes of the Zarr array.
        """
        if self._zarrArray.chunks:
            return self._zarrArray.chunks
        return "contiguous"

    @property
    def dimensionality(self):
        """ String that describes the dimensions of the underlying array.
        """
        return "array"

    @property
    def elementTypeName(self):
        """ String representation of the array's data type.
        """
        dtype = self._zarrArray.dtype
        return str(dtype)

    @property
    def attributes(self):
        """ Returns a dictionary with the Zarr array's attributes.
        """
        return attrsToDict(self._zarrArray.attrs)

    @property
    def dimensionNames(self):
        """ Returns a list of dimension names.

            Looks for '_ARRAY_DIMENSIONS' attribute (xarray convention),
            otherwise generates default names.
        """
        attrs = self._zarrArray.attrs

        # Check for xarray/NetCDF dimension names convention
        if '_ARRAY_DIMENSIONS' in attrs:
            return list(attrs['_ARRAY_DIMENSIONS'])

        # Generate default dimension names
        return [DIM_TEMPLATE.format(dimNr) for dimNr in range(len(self._zarrArray.shape))]

    @property
    def unit(self):
        """ Returns the unit of the data.
        """
        return zarrArrayUnit(self._zarrArray)

    @property
    def missingDataValue(self):
        """ Returns the value used to indicate missing data.
        """
        return zarrArrayMissingValue(self._zarrArray)

    @property
    def summary(self):
        """ Returns a summary string describing the array shape.
        """
        return shapeToSummary(self._zarrArray.shape)

    def _fetchAllChildren(self):
        """ Fetches children for structured arrays (fields).
        """
        from argos.repo.memoryrtis import FieldRti

        childItems = []

        if self._isStructured:
            fieldNames = self._zarrArray.dtype.names
            for fieldName in fieldNames:
                childItems.append(FieldRti(
                    self._zarrArray[:],  # Load full array for field access
                    nodeName=fieldName,
                    fileName=self.fileName,
                    iconColor=self.iconColor
                ))

        return childItems


class ZarrGroupRti(BaseRti):
    """ Repository Tree Item (RTI) that contains a Zarr group.

        Can represent both the root of a Zarr store (file-level) or
        a nested group within the hierarchy.
    """
    _defaultIconGlyph = RtiIconFactory.FOLDER

    def __init__(self, zarrGroup=None, nodeName='', fileName='',
                 iconColor=ICON_COLOR_UNDEF, isRoot=False):
        """ Constructor.

            :param zarrGroup: the underlying Zarr group (None for file-level RTI)
            :param nodeName: name of this node in the tree
            :param fileName: path to the Zarr store
            :param iconColor: icon color for this item
            :param isRoot: True if this is the root/file-level RTI
        """
        super(ZarrGroupRti, self).__init__(
            nodeName=nodeName,
            fileName=fileName,
            iconColor=iconColor
        )
        self._zarrGroup = zarrGroup
        self._isRoot = isRoot

        if isRoot:
            self._checkFileExists()

    @property
    def _defaultIconGlyph(self):
        """ Returns FILE icon for root, FOLDER for nested groups.
        """
        if self._isRoot:
            return RtiIconFactory.FILE
        return RtiIconFactory.FOLDER

    def hasChildren(self):
        """ Returns True - groups can contain arrays and subgroups.
        """
        return True

    @property
    def attributes(self):
        """ Returns a dictionary with the group's attributes.
        """
        if self._zarrGroup is not None:
            return attrsToDict(self._zarrGroup.attrs)
        return {}

    @property
    def summary(self):
        """ Returns a summary of the group contents.
        """
        if self._zarrGroup is not None:
            try:
                count = len(list(self._zarrGroup.keys()))
                return "{} item{}".format(count, 's' if count != 1 else '')
            except Exception:
                pass
        return ""

    def _openResources(self):
        """ Opens the Zarr store for reading.

            Only opens the store for root/file-level RTIs.
            Child groups already have their zarr group reference set.
        """
        import zarr

        # Only open the store if this is the root and we don't have a group yet
        if self._isRoot and self._zarrGroup is None:
            logger.info("Opening Zarr store: {}".format(self._fileName))

            if not os.path.exists(self._fileName):
                raise OSError("Zarr store does not exist: {}".format(self._fileName))

            self._zarrGroup = zarr.open(self._fileName, mode='r')

    def _closeResources(self):
        """ Closes the Zarr store and releases resources.

            Only clears resources for root/file-level RTIs.
        """
        if self._isRoot:
            logger.info("Closing Zarr store: {}".format(self._fileName))
        self._zarrGroup = None

    def _fetchAllChildren(self):
        """ Fetches all child items (arrays and subgroups).
        """
        import zarr

        childItems = []

        if self._zarrGroup is None:
            return childItems

        try:
            # Sort keys for consistent ordering
            for key in sorted(self._zarrGroup.keys()):
                try:
                    item = self._zarrGroup[key]

                    if isinstance(item, zarr.Array):
                        child = ZarrArrayRti(
                            zarrArray=item,
                            nodeName=key,
                            fileName=self.fileName,
                            iconColor=self.iconColor
                        )
                    elif isinstance(item, zarr.Group):
                        child = ZarrGroupRti(
                            zarrGroup=item,
                            nodeName=key,
                            fileName=self.fileName,
                            iconColor=self.iconColor,
                            isRoot=False
                        )
                    else:
                        logger.warning("Skipping unknown Zarr item type: {} ({})"
                                       .format(key, type(item)))
                        continue

                    childItems.append(child)

                except Exception as ex:
                    logger.warning("Error reading Zarr item '{}': {}".format(key, ex))

        except Exception as ex:
            logger.error("Error iterating Zarr group: {}".format(ex))

        return childItems


class ZarrFileRti(ZarrGroupRti):
    """ Repository Tree Item for a Zarr store (file/directory level).

        This is the entry point when opening a .zarr directory.
    """
    _defaultIconGlyph = RtiIconFactory.FILE

    def __init__(self, nodeName='', fileName='', iconColor=ICON_COLOR_UNDEF):
        """ Constructor.

            :param nodeName: name of this node in the tree
            :param fileName: path to the Zarr store
            :param iconColor: icon color for this item
        """
        super(ZarrFileRti, self).__init__(
            zarrGroup=None,
            nodeName=nodeName,
            fileName=fileName,
            iconColor=iconColor,
            isRoot=True
        )

    @classmethod
    def createFromFileName(cls, fileName, iconColor=ICON_COLOR_UNDEF):
        """ Factory method to create a ZarrFileRti from a file path.

            :param fileName: path to the Zarr store
            :param iconColor: icon color for this item
            :return: ZarrFileRti instance
        """
        # Use the base name of the path as the node name
        nodeName = os.path.basename(fileName)
        return cls(nodeName=nodeName, fileName=fileName, iconColor=iconColor)
