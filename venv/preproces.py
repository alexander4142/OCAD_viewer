# "C:\Program Files\OCAD\OCAD 2018 Viewer\Samples\Forest Orienteering Map Bürenflue.ocd"
# Define the binary format based on the Delphi structure
# SmallInt = h, byte = B, cardinal = I, integer = i, double = d, word = H
import struct
import numpy as np
import os


def parse_tdpoly(data):
    """Convert TDPoly-formatted int32 x and y into real coordinates and flags."""
    data = np.array(data)
    # Extract flags (lower 8 bits)
    x_flags = data[::2] & 0xFF
    y_flags = data[1::2] & 0xFF

    return data >> 8, x_flags, y_flags


def read_extra_symbol(data, start):
    header_format = "h H h h h h h h"
    header_size = struct.calcsize(header_format)
    us = struct.unpack(header_format, data[start:start + header_size])
    symbol = {"stType": us[0],
              "stFlags": us[1],
              "stColor": us[2],
              "stLineWidth": us[3],
              "stDiameter": us[4],
              "stnPoly": us[5],
              "stRes1": us[6],
              "stRes2": us[7],
              "stPoly": []}
    poly_format = " ".join("i i" for i in range(symbol['stnPoly']))
    poly_size = struct.calcsize(poly_format)
    us2 = struct.unpack(poly_format, data[start + header_size:start + header_size + poly_size])
    temp_data = parse_tdpoly(list(us2))
    symbol['stPoly'] = temp_data[0]
    symbol['x_flags'] = temp_data[1]
    symbol['y_flags'] = temp_data[2]

    return symbol


def read_header(filename):
    with open(filename, "rb") as f:
        header_format = "<h B B h"
        header_size = struct.calcsize(header_format)
        header_data = f.read(header_size)

        ocad_mark, file_type, file_status, version = struct.unpack(header_format, header_data)

        if version == 2018:
            return read_ocad_header_2018(filename)
        if version == 9:
            return read_ocad_header_9(filename)


def read_ocad_header_9(filename):
    """Reads the OCAD file header (60 bytes)."""
    with open(filename, "rb") as f:
        header_data = f.read(48)

        fields = struct.unpack("<h B B h h i i i i i i i i i i", header_data)

        header = {
            "OCADMark": fields[0],  # Should be 3245 (0x0CAD)
            "FileType": fields[1],
            "FileStatus": fields[2],
            "Version": fields[3],  # Example: 2018
            "Subversion": fields[4],
            "FirstSymbolIndexBlk": fields[5],
            "ObjectIndexBlock": fields[6],  # This is what we need next!
            "Res0": fields[7],
            "Res1": fields[8],
            "Res2": fields[9],
            "InfoSize": fields[10],
            "FirstStringIndexBlk": fields[11],
            "FileNamePos": fields[12],
            "FileNameSize": fields[13]
        }

    return header


def read_ocad_header_2018(filename):
    """Reads the OCAD file header (60 bytes)."""
    with open(filename, "rb") as f:
        header_data = f.read(60)  # Read first 60 bytes

        fields = struct.unpack("<h B B h B B I I i i I I I I I I I I I", header_data)

        header = {
            "OCADMark": fields[0],  # Should be 3245 (0x0CAD)
            "FileType": fields[1],
            "FileStatus": fields[2],
            "Version": fields[3],  # Example: 2018
            "Subversion": fields[4],
            "SubSubversion": fields[5],
            "FirstSymbolIndexBlk": fields[6],
            "ObjectIndexBlock": fields[7],  # This is what we need next!
            "OfflineSyncSerial": fields[8],
            "CurrentFileVersion": fields[9],
            "Internal1": fields[10],
            "Internal2": fields[11],
            "Internal3": fields[12],
            "FirstStringIndexBlk": fields[13],
            "FileNamePos": fields[14],
            "FileNameSize": fields[15],
            "Internal": fields[16],
            "Res1": fields[17],
            "Res2": fields[18]
        }

    return header


def reverse_bits_32(n):
    n &= 0xFFFFFFFF
    n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1)
    n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2)
    n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4)
    n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8)
    n = (n >> 16) | (n << 16)
    return n & 0xFFFFFFFF


def read_all_string_index_blocks(filename, start_pos):
    blocks = []
    with open(filename, "rb") as f:
        # --- Detect file version ---
        f.seek(0)
        header = f.read(32)  # more than enough for version field
        # In OCAD 9, version is often at offset 4; in OCAD 11+ it's still here but larger values
        version, = struct.unpack_from("<I", header, 4)
        # Set struct format depending on file version
        if version <= 10:
            # Old layout (OCAD 9, 10)
            # TStringIndexBlock: NextIndexBlock (4 bytes) + 256 TStringIndex
            # TStringIndex: Pos (4), Len (4), RecType (4), ObjIndex (4)
            index_struct = "<i i i i"  # Pos, Len, RecType, ObjIndex
            index_size = struct.calcsize(index_struct)
        else:
            # Newer layout (OCAD 11+)
            # TStringIndex: Pos (4), Len (4), RecType (4), ObjIndex (4)
            # but sometimes Len is unsigned, and strings are UTF-8
            index_struct = "<I i i i"
            index_size = struct.calcsize(index_struct)

        next_block_pos = start_pos

        while next_block_pos != 0:
            f.seek(next_block_pos)
            # Read NextIndexBlock
            block_header = f.read(4)
            # print(" ".join(f"{b:02X}" for b in block_header))
            if len(block_header) < 4:
                break  # Unexpected EOF
            next_block_pos = struct.unpack_from("<I", block_header)[0]

            table = []
            for _ in range(256):
                entry_bytes = f.read(index_size)
                if len(entry_bytes) < index_size:
                    break  # Incomplete block
                pos, length, rectype, objindex = struct.unpack_from(index_struct, entry_bytes)
                table.append((pos, (length), rectype, objindex))

            blocks.extend(table)

    return blocks


def read_all_string_index_blocks_9(filename, start_position):
    ENTRY_STRUCT = struct.Struct("<Iiii")  # Pos: uint32, Len: int32, RecType: int32, ObjIndex: int32
    BLOCK_HEADER_STRUCT = struct.Struct("<I")  # next_block pointer
    ENTRIES_PER_BLOCK = 256
    entries = []
    with open(filename, "rb") as f:
        position = start_position
        while position != 0:
            f.seek(position)
            block_data = f.read(BLOCK_HEADER_STRUCT.size + ENTRIES_PER_BLOCK * ENTRY_STRUCT.size)

            # Get pointer to next block
            next_block, = BLOCK_HEADER_STRUCT.unpack_from(block_data, 0)

            # Read all entries
            offset = BLOCK_HEADER_STRUCT.size
            for _ in range(ENTRIES_PER_BLOCK):
                pos, length, rectype, objindex = ENTRY_STRUCT.unpack_from(block_data, offset)
                offset += ENTRY_STRUCT.size
                if pos != 0:  # skip empty slots
                    entries.append({
                        "pos": pos,
                        "length": length,
                        "rectype": rectype,
                        "objindex": objindex
                    })

            position = next_block

    return entries


def read_ocad_string_9(f, pos, length):
    """
    Reads and decodes a single OCAD string from the file.
    Returns a list of (code, value) tuples, plus optional first field.
    """
    f.seek(pos)
    raw = f.read(abs(length))  # length is reversed for actual size sometimes
    raw = raw.split(b"\x00", 1)[0]  # stop at null terminator
    parts = raw.decode("utf-8", errors="ignore").split("\t")
    # ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[29:93])).strip('\x00').replace('\x00', '')

    # Has a first field
    first_field = parts[0]
    pairs = {}
    for part in parts[1:]:
        if len(part) > 1:
            pairs[part[0]] = part[1:]

    return first_field, pairs


def read_all_strings(filename, entries):
    out = []
    with open(filename, "rb") as f:
        for e in entries:
            if e[2] > 0:  # ignore deleted strings
                first_field, pairs = read_ocad_string_9(f, e[0], e[1])
                out.append((e[2], first_field, pairs))
    return out


def read_all_object_indices(filename, object_index_pos):
    """Reads all Object Index Blocks and returns a list of objects with metadata."""
    object_list = []
    TOBJECT_INDEX_FORMAT = "<4i i i i b b b b h h h B B"
    TOBJECT_INDEX_SIZE = 40

    with open(filename, "rb") as f:
        while object_index_pos != 0:  # Continue until no more blocks
            f.seek(object_index_pos)  # Jump to block position

            data = f.read(10296)  # Read full Object Index Block

            if len(data) < 10296:
                print(f"Error: Expected 10296 bytes, but read {len(data)}")
                break  # Stop if not enough data

            # Unpack first 4 bytes (NextObjectIndexBlock)
            next_block = struct.unpack_from("<i", data, 0)[0]

            # Read 256 TObjectIndex entries (40 bytes each)
            for i in range(256):
                offset = 4 + (i * TOBJECT_INDEX_SIZE)
                if offset + TOBJECT_INDEX_SIZE > len(data):
                    break  # Stop if not enough data for a full record

                # Unpack TObjectIndex (40 bytes)
                obj = struct.unpack_from(TOBJECT_INDEX_FORMAT, data, offset)

                # Convert to dictionary
                object_list.append({
                    "BoundingBox": np.array(obj[:4]) >> 8,  # (llx, lly, urx, ury)
                    "Pos": obj[4],
                    "Len": obj[5],
                    "Sym": obj[6],
                    "ObjType": obj[7],
                    "EncryptedMode": obj[8],
                    "Status": obj[9],
                    "ViewType": obj[10],
                    "Color": obj[11],
                    "Group": obj[12],
                    "ImpLayer": obj[13],
                    "DbDatasetHash": obj[14],
                    "DbKeyHash": obj[15],
                })

            # Move to next block
            object_index_pos = next_block
        print("Ocad indices read. ", len(object_list), " objects.")
    return object_list


def read_ocad_object_9(f, pos):
    TOCAD_OBJECT_FORMAT = "<i B B h i h h i h h d d"
    TOCAD_OBJECT_SIZE = 40

    f.seek(pos)

    data = f.read(TOCAD_OBJECT_SIZE)
    if len(data) < TOCAD_OBJECT_SIZE:
        print(f"Error: Incomplete object at position {pos}")
        return None

    # Unpack structure
    objs = struct.unpack(TOCAD_OBJECT_FORMAT, data)

    # Convert to dictionary
    ocad_object = {
        "Sym": objs[0],
        "Otp": objs[1],
        "Res0": objs[2],
        "Ang": objs[3],
        "nItem": objs[4],
        "nText": objs[5],
        "Res1": objs[6],
        "Col": objs[7],
        "LineWidth": objs[8],
        "DiamFlags": objs[9],
        "Res2": objs[10],
        "Res3": objs[11],
        "Coordinates": []
    }

    # Read coordinates (TDPoly) if nItem > 0
    if ocad_object["nItem"] > 0:
        coord_format = f"<{ocad_object['nItem'] * 2}i"  # Each coordinate = 2 integers
        coord_size = struct.calcsize(coord_format)
        coord_data = f.read(coord_size)

        if len(coord_data) == coord_size:
            temp_data = parse_tdpoly(list(struct.unpack(coord_format, coord_data)))
            ocad_object['Coordinates'] = temp_data[0].reshape(-1, 2)
            ocad_object['x_flags'] = temp_data[1]
            ocad_object['y_flags'] = temp_data[2]

    return ocad_object


def read_ocad_object(f, pos):
    """Reads a TOcadObject from the given position in the file."""
    TOCAD_OBJECT_FORMAT = "<i B B h i h h i i d I d I H H H B B"
    TOCAD_OBJECT_SIZE = 56

    f.seek(pos)  # Jump to object position

    # Read fixed TOcadObject fields (32 bytes)
    data = f.read(TOCAD_OBJECT_SIZE)
    if len(data) < TOCAD_OBJECT_SIZE:
        print(f"Error: Incomplete object at position {pos}")
        return None

    # Unpack structure
    objs = struct.unpack(TOCAD_OBJECT_FORMAT, data)

    # Convert to dictionary
    ocad_object = {
        "Sym": objs[0],
        "Otp": objs[1],
        "_Customer": objs[2],
        "Ang": objs[3],
        "Col": objs[4],
        "LineWidth": objs[5],
        "DiamFlags": objs[6],
        "ServerObjectId": objs[7],
        "Height": objs[8],
        "CreationDate": objs[9],
        "MultirepresentationId": objs[10],
        "ModificationDate": objs[11],
        "nItem": objs[12],  # Number of coordinates
        "nText": objs[13],  # Number of text characters
        "nObjectString": objs[14],
        "nDatabaseString": objs[15],
        "ObjectStringType": objs[16],
        "Res1": objs[17],
        "Coordinates": [],
        "Text": "",
    }

    # Read coordinates (TDPoly) if nItem > 0
    if ocad_object["nItem"] > 0:
        coord_format = f"<{ocad_object['nItem'] * 2}i"  # Each coordinate = 2 integers
        coord_size = struct.calcsize(coord_format)
        coord_data = f.read(coord_size)

        if len(coord_data) == coord_size:
            temp_data = parse_tdpoly(list(struct.unpack(coord_format, coord_data)))
            ocad_object['Coordinates'] = temp_data[0].reshape(-1, 2)
            ocad_object['x_flags'] = temp_data[1]
            ocad_object['y_flags'] = temp_data[2]

    # Read text if nText > 0
    if ocad_object["nText"] > 0:
        text_data = f.read(ocad_object["nText"])
        ocad_object["Text"] = text_data.decode("utf-8", errors="ignore").strip("\x00")
    return ocad_object


def read_all_ocad_objects(filename, ol, version=2018):
    """Reads all TOcadObjects based on the object list."""
    out = []

    with open(filename, "rb") as f:
        for objs in ol:
            ocad_obj = read_ocad_object(f, objs["Pos"]) if version == 2018 else read_ocad_object_9(f, objs["Pos"])
            if ocad_obj and objs['Status'] == 1:
                out.append(ocad_obj)
    print("All objects read")
    return out


def read_symbol_index_blocks(file_path, first_symbol_index_blk):
    symbol_blocks = []

    with open(file_path, 'rb') as f:
        current_block_pos = first_symbol_index_blk

        while current_block_pos != 0:
            f.seek(current_block_pos)

            # Read the block (1028 bytes)
            data = f.read(1028)
            if len(data) != 1028:
                break  # Corrupt or incomplete file

            # Unpack the block
            next_block, *symbol_positions = struct.unpack('<I256i', data)

            # Store the block
            symbol_blocks.append({
                'NextSymbolIndexBlock': next_block,
                'SymbolPositions': [pos for pos in symbol_positions if pos > 0]  # Ignore 0 values
            })

            # Move to the next block
            current_block_pos = next_block

    return symbol_blocks


def read_symbols(file_path, symbol_index_blocks):
    symbols = {}
    with open(file_path, "rb") as f:
        for block in symbol_index_blocks:
            for pos in block["SymbolPositions"]:
                if pos > 0:
                    f.seek(pos)
                    size_data = f.read(9)
                    if len(size_data) < 9:
                        continue
                    size, symNum, otp = struct.unpack("i i B", size_data)

                    if otp == 1:
                        f.seek(pos)
                        symbol_data = f.read(size)
                        point_format = "i i B B ? B B B B B i I B B h 14h 64c 548B 64H H h"
                        point_sym_size = struct.calcsize(point_format)
                        unpacked_symbol = struct.unpack(
                            point_format,
                            symbol_data[:point_sym_size]
                        )
                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "notUsed1": unpacked_symbol[12],
                            "notUsed2": unpacked_symbol[13],
                            "nColors": unpacked_symbol[14],
                            "Colors": unpacked_symbol[15:29],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[29:93])).strip('\x00').replace('\x00', ''),
                            "IconBits": list(unpacked_symbol[93:641]),
                            "SymbolTreeGroup": unpacked_symbol[641:705],
                            "DataSize": unpacked_symbol[705],
                            "Reserved": unpacked_symbol[706],
                            "Elements": []
                        }
                        totalUnits = formatted_symbol['DataSize']
                        processed_units = 0
                        element_offset = point_sym_size

                        lf = "h H h h h h h h"
                        unit_size = struct.calcsize(lf)
                        while processed_units < totalUnits:
                            us = struct.unpack(lf, symbol_data[element_offset:element_offset + unit_size])

                            ss = {"stType": us[0],
                                  "stFlags": us[1],
                                  "stColor": us[2],
                                  "stLineWidth": us[3],
                                  "stDiameter": us[4],
                                  "stnPoly": us[5],
                                  "stRes1": us[6],
                                  "stRes2": us[7],
                                  "stPoly": []}
                            lf2 = " ".join(["i i" for i in range(ss['stnPoly'])])
                            poly_size = struct.calcsize(lf2)
                            us2 = struct.unpack(lf2, symbol_data[element_offset + unit_size:element_offset + unit_size + poly_size])
                            temp_data = parse_tdpoly(list(us2))
                            ss['stPoly'] = temp_data[0].reshape(-1, 2)
                            ss['x_flags'] = temp_data[1]
                            ss['y_flags'] = temp_data[2]
                            processed_units += 2 + poly_size / 8
                            element_offset += unit_size + poly_size
                            formatted_symbol['Elements'].append(ss)
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol

                    if otp == 2: # Line symbol processing
                        f.seek(pos)
                        symbol_data = f.read(size)
                        line_format = "i i B B ? B B B B B i I B B h 14h 64c 548B 64H h h h h h h h h h h h h h H H h h h h h h h h h 2h H h B B h h h H H H H H B B"
                        line_size = struct.calcsize(line_format)
                        unpacked_symbol = struct.unpack(
                            line_format,
                            symbol_data[:line_size]
                        )

                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "notUsed1": unpacked_symbol[12],
                            "notUsed2": unpacked_symbol[13],
                            "nColors": unpacked_symbol[14],
                            "Colors": unpacked_symbol[15:29],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[29:93])).strip('\x00').replace('\x00', ''),
                            "IconBits": list(unpacked_symbol[93:641]),
                            "SymbolTreeGroup": unpacked_symbol[641:705],
                            "LineColor": unpacked_symbol[705],
                            "LineWidth": unpacked_symbol[706],
                            "LineStyle": unpacked_symbol[707],
                            "DistFromStart": unpacked_symbol[708],
                            "DistToEnd": unpacked_symbol[709],
                            "MainLength": unpacked_symbol[710],
                            "EndLength": unpacked_symbol[711],
                            "MainGap": unpacked_symbol[712],
                            "SecGap": unpacked_symbol[713],
                            "EndGap": unpacked_symbol[714],
                            "MinSym": unpacked_symbol[715],
                            "nPrimSym": unpacked_symbol[716],
                            "PrimSymDist": unpacked_symbol[717],
                            "DblMode": unpacked_symbol[718],
                            "DblFlags": unpacked_symbol[719],
                            "DblFillColor": unpacked_symbol[720],
                            "DblLeftColor": unpacked_symbol[721],
                            "DblRightColor": unpacked_symbol[722],
                            "DblWidth": unpacked_symbol[723],
                            "DblLeftWidth": unpacked_symbol[724],
                            "DblRightWidth": unpacked_symbol[725],
                            "DblLength": unpacked_symbol[726],
                            "DblGap": unpacked_symbol[727],
                            "DblBackgroundColor": unpacked_symbol[728],
                            "DblRes": unpacked_symbol[729:731],
                            "DecMode": unpacked_symbol[731],
                            "DecSymbolSize": unpacked_symbol[732],
                            "DecSymbolDistance": unpacked_symbol[733],
                            "DecSymbolWidth": unpacked_symbol[734],
                            "FrColor": unpacked_symbol[735],
                            "FrWidth": unpacked_symbol[736],
                            "FrStyle": unpacked_symbol[737],
                            "PrimDSize": unpacked_symbol[738],
                            "SecDSize": unpacked_symbol[739],
                            "CornerDSize": unpacked_symbol[740],
                            "StartDSize": unpacked_symbol[741],
                            "EndDSize": unpacked_symbol[742],
                            "UseSymbolFlags": unpacked_symbol[743],
                            "Reserved": unpacked_symbol[744],
                            "Symbols": {}
                        }
                        offset = 0
                        if formatted_symbol['PrimDSize']:
                            formatted_symbol['Symbols']['Main'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['PrimDSize'] * 8
                        if formatted_symbol['SecDSize']:
                            formatted_symbol['Symbols']['Secondary'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['SecDSize'] * 8
                        if formatted_symbol['CornerDSize']:
                            formatted_symbol['Symbols']['Corner'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['CornerDSize'] * 8
                        if formatted_symbol['StartDSize']:
                            formatted_symbol['Symbols']['Start'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['StartDSize'] * 8
                        if formatted_symbol['EndDSize']:
                            formatted_symbol['Symbols']['End'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['EndDSize'] * 8
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol

                    if otp == 3:  # Area Symbol Processing
                        f.seek(pos)
                        symbol_data = f.read(size)
                        area_format = "i i B B ? B B B B B i I B B h 14h 64c 548B 64H i h h h h h h h ? ? B B h h h B B h h H"
                        area_size = struct.calcsize(area_format)
                        unpacked_symbol = struct.unpack(
                            area_format,
                            symbol_data[:area_size]
                        )

                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "notUsed1": unpacked_symbol[12],
                            "notUsed2": unpacked_symbol[13],
                            "nColors": unpacked_symbol[14],
                            "Colors": unpacked_symbol[15:29],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[29:93])).strip('\x00').replace('\x00', ''),
                            "IconBits": list(unpacked_symbol[93:641]),
                            "SymbolTreeGroup": unpacked_symbol[641:705],
                            "BorderSym": unpacked_symbol[705],
                            "FillColor": unpacked_symbol[706],
                            "HatchMode": unpacked_symbol[707],
                            "HatchColor": unpacked_symbol[708],
                            "HatchLineWidth": unpacked_symbol[709],
                            "HatchDist": unpacked_symbol[710],
                            "HatchAngle1": unpacked_symbol[711],
                            "HatchAngle2": unpacked_symbol[712],
                            "FillOn": unpacked_symbol[713],
                            "BorderOn": unpacked_symbol[714],
                            "StructMode": unpacked_symbol[715],
                            "StructDraw": unpacked_symbol[716],
                            "StructWidth": unpacked_symbol[717],
                            "StructHeight": unpacked_symbol[718],
                            "StructAngle": unpacked_symbol[719],
                            "StructIrregularVarX": unpacked_symbol[720],
                            "StructIrregularVarY": unpacked_symbol[721],
                            "StructIrregularMinDist": unpacked_symbol[722],
                            "StructRes": unpacked_symbol[723],
                            "DataSize": unpacked_symbol[724],
                            "Elements": []
                        }
                        totalUnits = formatted_symbol['DataSize']
                        processed_units = 0
                        element_offset = area_size

                        lf = "h H h h h h h h"
                        unit_size = struct.calcsize(lf)
                        while processed_units < totalUnits:
                            us = struct.unpack(lf, symbol_data[element_offset:element_offset + unit_size])

                            ss = {"stType": us[0],
                                  "stFlags": us[1],
                                  "stColor": us[2],
                                  "stLineWidth": us[3],
                                  "stDiameter": us[4],
                                  "stnPoly": us[5],
                                  "stRes1": us[6],
                                  "stRes2": us[7],
                                  "stPoly": []}
                            lf2 = " ".join(["i i" for i in range(ss['stnPoly'])])
                            poly_size = struct.calcsize(lf2)
                            us2 = struct.unpack(lf2, symbol_data[element_offset + unit_size:element_offset + unit_size + poly_size])
                            temp_data = parse_tdpoly(list(us2))
                            ss['stPoly'] = temp_data[0]
                            ss['x_flags'] = temp_data[1]
                            ss['y_flags'] = temp_data[2]
                            processed_units += 2 + poly_size / 8
                            element_offset += unit_size + poly_size
                            formatted_symbol['Elements'].append(ss)
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol

                        if otp == 7:  # Rectangle Symbol Processing
                            f.seek(pos)
                        symbol_data = f.read(size)
                        rect_format = "i i B B ? B B B B B i I B B h 14h 64c 484B 64H i h h h h h h h h ? ? B B h h h B B h h H"
                        unpacked_symbol = struct.unpack(
                            rect_format,
                            symbol_data[:struct.calcsize(rect_format)]
                        )

                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "nColors": unpacked_symbol[14],
                            "Colors": unpacked_symbol[15:29],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[29:93])).strip('\x00').replace('\x00', ''),
                            "IconBits": unpacked_symbol[93:577],
                            "SymbolTreeGroup": unpacked_symbol[577:641],
                            "LineColor": unpacked_symbol[641],
                            "LineWidth": unpacked_symbol[642],
                            "Radius": unpacked_symbol[643],
                            "GridFlags": unpacked_symbol[644],
                            "CellWidth": unpacked_symbol[645],
                            "CellHeight": unpacked_symbol[646],
                            "ResGridLineColor": unpacked_symbol[647],
                            "ResGridLineWidth": unpacked_symbol[648],
                            "FillOn": unpacked_symbol[649],
                            "UnnumCells": unpacked_symbol[650],
                            "UnnumText": unpacked_symbol[651],
                            "LineStyle": unpacked_symbol[652],
                            "Res2": unpacked_symbol[653],
                            "ResFontColor": unpacked_symbol[654],
                            "FontSize": unpacked_symbol[655],
                            "Res3": unpacked_symbol[656],
                            "Res4": unpacked_symbol[657],
                            "Res5": unpacked_symbol[658],
                            "Res6": unpacked_symbol[659]
                        }
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol
    return symbols

def read_symbols_9(file_path, symbol_index_blocks):
    symbols = {}
    with open(file_path, "rb") as f:
        for block in symbol_index_blocks:
            for pos in block["SymbolPositions"]:
                if pos > 0:
                    f.seek(pos)
                    size_data = f.read(9)
                    if len(size_data) < 9:
                        continue
                    size, symNum, otp = struct.unpack("i i B", size_data)

                    if otp == 1:
                        f.seek(pos)
                        symbol_data = f.read(size)
                        point_format = "i i B B ? B B B B B i i h h 14h 32c 484B H h"
                        point_sym_size = struct.calcsize(point_format)
                        unpacked_symbol = struct.unpack(
                            point_format,
                            symbol_data[:point_sym_size]
                        )
                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "Group": unpacked_symbol[12],
                            "nColors": unpacked_symbol[13],
                            "Colors": unpacked_symbol[14:28],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[28:60])).strip('\x00').replace('\x00', ''),
                            "IconBits": list(unpacked_symbol[60:544]),
                            "DataSize": unpacked_symbol[544],
                            "Reserved": unpacked_symbol[545],
                            "Elements": []
                        }
                        totalUnits = formatted_symbol['DataSize']
                        processed_units = 0
                        element_offset = point_sym_size

                        lf = "h H h h h h h h"
                        unit_size = struct.calcsize(lf)
                        while processed_units < totalUnits:
                            us = struct.unpack(lf, symbol_data[element_offset:element_offset + unit_size])

                            ss = {"stType": us[0],
                                  "stFlags": us[1],
                                  "stColor": us[2],
                                  "stLineWidth": us[3],
                                  "stDiameter": us[4],
                                  "stnPoly": us[5],
                                  "stRes1": us[6],
                                  "stRes2": us[7],
                                  "stPoly": []}
                            lf2 = " ".join(["i i" for i in range(ss['stnPoly'])])
                            poly_size = struct.calcsize(lf2)
                            us2 = struct.unpack(lf2, symbol_data[element_offset + unit_size:element_offset + unit_size + poly_size])
                            temp_data = parse_tdpoly(list(us2))
                            ss['stPoly'] = temp_data[0]
                            ss['x_flags'] = temp_data[1]
                            ss['y_flags'] = temp_data[2]
                            processed_units += 2 + poly_size / 8
                            element_offset += unit_size + poly_size
                            formatted_symbol['Elements'].append(ss)
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol

                    if otp == 2: # Line symbol processing
                        f.seek(pos)
                        symbol_data = f.read(size)
                        line_format = "i i B B ? B B B B B i i h h 14h 32c 484B h h h h h h h h h h h h h H H h h h h h h h h h 2h H h h h h h H H H H H h"
                        line_size = struct.calcsize(line_format)
                        unpacked_symbol = struct.unpack(
                            line_format,
                            symbol_data[:line_size]
                        )

                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "Group": unpacked_symbol[12],
                            "nColors": unpacked_symbol[13],
                            "Colors": unpacked_symbol[14:28],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[28:60])).strip('\x00').replace('\x00', ''),
                            "IconBits": list(unpacked_symbol[60:544]),
                            "LineColor": unpacked_symbol[544],
                            "LineWidth": unpacked_symbol[545],
                            "LineStyle": unpacked_symbol[546],
                            "DistFromStart": unpacked_symbol[547],
                            "DistToEnd": unpacked_symbol[548],
                            "MainLength": unpacked_symbol[549],
                            "EndLength": unpacked_symbol[550],
                            "MainGap": unpacked_symbol[551],
                            "SecGap": unpacked_symbol[552],
                            "EndGap": unpacked_symbol[553],
                            "MinSym": unpacked_symbol[554],
                            "nPrimSym": unpacked_symbol[555],
                            "PrimSymDist": unpacked_symbol[556],
                            "DblMode": unpacked_symbol[557],
                            "DblFlags": unpacked_symbol[558],
                            "DblFillColor": unpacked_symbol[559],
                            "DblLeftColor": unpacked_symbol[560],
                            "DblRightColor": unpacked_symbol[561],
                            "DblWidth": unpacked_symbol[562],
                            "DblLeftWidth": unpacked_symbol[563],
                            "DblRightWidth": unpacked_symbol[564],
                            "DblLength": unpacked_symbol[565],
                            "DblGap": unpacked_symbol[566],
                            "Res0": unpacked_symbol[567],
                            "Res1": unpacked_symbol[568:570],
                            "DecMode": unpacked_symbol[570],
                            "DecLast": unpacked_symbol[571],
                            "Res": unpacked_symbol[572],
                            "FrColor": unpacked_symbol[573],
                            "FrWidth": unpacked_symbol[574],
                            "FrStyle": unpacked_symbol[575],
                            "PrimDSize": unpacked_symbol[576],
                            "SecDSize": unpacked_symbol[577],
                            "CornerDSize": unpacked_symbol[578],
                            "StartDSize": unpacked_symbol[579],
                            "EndDSize": unpacked_symbol[580],
                            "Reserved": unpacked_symbol[581],
                            "Symbols": {}
                        }
                        offset = 0
                        if formatted_symbol['PrimDSize']:
                            formatted_symbol['Symbols']['Main'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['PrimDSize'] * 8
                        if formatted_symbol['SecDSize']:
                            formatted_symbol['Symbols']['Secondary'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['SecDSize'] * 8
                        if formatted_symbol['CornerDSize']:
                            formatted_symbol['Symbols']['Corner'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['CornerDSize'] * 8
                        if formatted_symbol['StartDSize']:
                            formatted_symbol['Symbols']['Start'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['StartDSize'] * 8
                        if formatted_symbol['EndDSize']:
                            formatted_symbol['Symbols']['End'] = read_extra_symbol(symbol_data, line_size + offset)
                            offset += formatted_symbol['EndDSize'] * 8
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol
                    if otp == 3:  # Area Symbol Processing
                        f.seek(pos)
                        symbol_data = f.read(size)
                        area_format = "i i B B ? B B B B B i i h h 14h 32c 484B i h h h h h h h ? ? h h h h h H"
                        area_size = struct.calcsize(area_format)
                        unpacked_symbol = struct.unpack(
                            area_format,
                            symbol_data[:area_size]
                        )

                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "Group": unpacked_symbol[12],
                            "nColors": unpacked_symbol[13],
                            "Colors": unpacked_symbol[14:28],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[28:60])).strip('\x00').replace('\x00', ''),
                            "IconBits": list(unpacked_symbol[60:544]),
                            "BorderSym": unpacked_symbol[544],
                            "FillColor": unpacked_symbol[545],
                            "HatchMode": unpacked_symbol[546],
                            "HatchColor": unpacked_symbol[547],
                            "HatchLineWidth": unpacked_symbol[548],
                            "HatchDist": unpacked_symbol[549],
                            "HatchAngle1": unpacked_symbol[550],
                            "HatchAngle2": unpacked_symbol[551],
                            "FillOn": unpacked_symbol[552],
                            "BorderOn": unpacked_symbol[553],
                            "StructMode": unpacked_symbol[554],
                            "StructWidth": unpacked_symbol[555],
                            "StructHeight": unpacked_symbol[556],
                            "StructAngle": unpacked_symbol[557],
                            "Res": unpacked_symbol[558],
                            "DataSize": unpacked_symbol[559],
                            "Elements": []
                        }
                        totalUnits = formatted_symbol['DataSize']
                        processed_units = 0
                        element_offset = area_size

                        lf = "h H h h h h h h"
                        unit_size = struct.calcsize(lf)
                        while processed_units < totalUnits:
                            us = struct.unpack(lf, symbol_data[element_offset:element_offset + unit_size])

                            ss = {"stType": us[0],
                                  "stFlags": us[1],
                                  "stColor": us[2],
                                  "stLineWidth": us[3],
                                  "stDiameter": us[4],
                                  "stnPoly": us[5],
                                  "stRes1": us[6],
                                  "stRes2": us[7],
                                  "stPoly": []}
                            lf2 = " ".join(["i i" for i in range(ss['stnPoly'])])
                            poly_size = struct.calcsize(lf2)
                            us2 = struct.unpack(lf2, symbol_data[element_offset + unit_size:element_offset + unit_size + poly_size])
                            temp_data = parse_tdpoly(list(us2))
                            ss['stPoly'] = temp_data[0]
                            ss['x_flags'] = temp_data[1]
                            ss['y_flags'] = temp_data[2]
                            processed_units += 2 + poly_size / 8
                            element_offset += unit_size + poly_size
                            formatted_symbol['Elements'].append(ss)
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol
                        continue
                        if otp == 7:  # Rectangle Symbol Processing
                            f.seek(pos)
                        symbol_data = f.read(size)
                        rect_format = "i i B B ? B B B B B i I B B h 14h 64c 484B 64H i h h h h h h h h ? ? B B h h h B B h h H"
                        unpacked_symbol = struct.unpack(
                            rect_format,
                            symbol_data[:struct.calcsize(rect_format)]
                        )

                        formatted_symbol = {
                            "Size": unpacked_symbol[0],
                            "SymNum": unpacked_symbol[1],
                            "Otp": unpacked_symbol[2],
                            "Flags": unpacked_symbol[3],
                            "Selected": unpacked_symbol[4],
                            "Status": unpacked_symbol[5],
                            "PreferredDrawingTool": unpacked_symbol[6],
                            "CsMode": unpacked_symbol[7],
                            "CsObjType": unpacked_symbol[8],
                            "CsCDFlags": unpacked_symbol[9],
                            "Extent": unpacked_symbol[10],
                            "FilePos": unpacked_symbol[11],
                            "nColors": unpacked_symbol[14],
                            "Colors": unpacked_symbol[15:29],
                            "Description": ''.join(map(lambda c: c.decode('utf-8', 'ignore'), unpacked_symbol[29:93])).strip('\x00').replace('\x00', ''),
                            "IconBits": unpacked_symbol[93:577],
                            "SymbolTreeGroup": unpacked_symbol[577:641],
                            "LineColor": unpacked_symbol[641],
                            "LineWidth": unpacked_symbol[642],
                            "Radius": unpacked_symbol[643],
                            "GridFlags": unpacked_symbol[644],
                            "CellWidth": unpacked_symbol[645],
                            "CellHeight": unpacked_symbol[646],
                            "ResGridLineColor": unpacked_symbol[647],
                            "ResGridLineWidth": unpacked_symbol[648],
                            "FillOn": unpacked_symbol[649],
                            "UnnumCells": unpacked_symbol[650],
                            "UnnumText": unpacked_symbol[651],
                            "LineStyle": unpacked_symbol[652],
                            "Res2": unpacked_symbol[653],
                            "ResFontColor": unpacked_symbol[654],
                            "FontSize": unpacked_symbol[655],
                            "Res3": unpacked_symbol[656],
                            "Res4": unpacked_symbol[657],
                            "Res5": unpacked_symbol[658],
                            "Res6": unpacked_symbol[659]
                        }
                        symbols[formatted_symbol["SymNum"]] = formatted_symbol
    return symbols
