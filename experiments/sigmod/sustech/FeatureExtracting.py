import re

import pandas as pd
import skrub

import sempipes


def extract_x2(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean X2.csv data to a readable format.

    :param data: X2.csv

    :return:
        A DataFrame which contains following columns:
        {instance_id: instance_id of items;
         brand: item's brand, for example: {'intenso', 'pny', 'lexar'}
         capacity: usb/sd card's capacity, unit in GB
         price: price of the item
         mem_type: memory type, for example: {'ssd', 'sd', 'microsd', 'usb'}
         type: type information, relative to brand
         model: model information, relative to brand
         item_code: the unique item code
         title: title information of instance}

         if the value can't extract from the information given, '0' will be filled.
    """
    brands = ["sandisk", "lexar", "kingston", "intenso", "toshiba", "sony", "pny", "samsung", "transcend"]
    families = {
        "sandisk": [
            "cruizer",
            "tarjeta",
            "glide",
            "select",
            "extern",
            "origin",
            "transmemory",
            "react",
            "memo",
            "kart",
            "pendrive",
            "car",
            "serie",
            "line",
            "extreme",
            "cruzer",
            "ultra",
            "micro",
            "traveler",
            "hyperx",
            "adapt",
            "wex",
            "flash",
        ],
        "lexar": [
            "ultra",
            "xqd",
            "jumpdrive",
            "micro",
            "pendrive",
            "sd",
            "tarjeta",
            "jumpdrive",
            "usb",
            "memo",
            "extreme",
            "blade",
            "car",
            "scheda",
            "veloc",
            "react",
            "adapt",
            "secure",
            "premium",
            "wex",
            "transmemo",
            "alu",
            "datatravel",
            "canvas",
            "flair",
            "hyperx",
            "cruzer",
            "flash",
        ],
        "toshiba": [
            "ultra",
            "exceria",
            "traveler",
            "sdhc",
            "memoria",
            "xqd",
            "line",
            "usb",
            "exceria",
            "transmemo",
            "extreme",
            "flair",
            "micro",
            "speicher",
            "serie",
            "car",
        ],
        "kingston": [
            "traveler",
            "cart",
            "adapt",
            "extreme",
            "memo",
            "canvas",
            "datatravel",
            "hyperx",
            "kart",
            "blade",
            "ultimate",
        ],
        "sony": [
            "extreme",
            "usm32gqx",
            "micro",
            "sd",
            "usb",
            "ultra",
            "jumpdrive",
            "hyperx",
            "memo",
            "kart",
            "xqd",
            "pendrive",
            "adapt",
            "blade",
            "cruzer",
            "flair",
            "glide",
            "cart",
            "tarjeta",
            "flash",
        ],
        "intenso": [
            "cs/ultra",
            "premium",
            "ultra",
            "micro",
            "micro",
            "line",
            "scheda",
            "usb",
            "sd",
            "premium",
            "tarjeta",
            "kart",
            "car",
            "transmemo",
        ],
        "pny": ["attach", "usb", "sd", "micro", "premium", "memo"],
        "samsung": [
            "galaxy",
            "speicher",
            "micro",
            "usb",
            "sd",
            "evo",
            "ultra",
            "extreme",
            "memo",
            "adapt",
            "car",
            "kart",
            "klasse",
            "multi",
            "jumpdrive",
            "flash",
        ],
        "transcend": [],
        "0": [
            "adapt",
            "alu",
            "attach",
            "blade",
            "canvas",
            "cart",
            "cruzer",
            "cs/ultra",
            "datatravel",
            "evo",
            "exceria",
            "extern",
            "extreme",
            "flair",
            "flash",
            "galaxy",
            "glide",
            "hyperx",
            "jumpdrive",
            "kart",
            "klasse",
            "line",
            "memo",
            "memoria",
            "multi",
            "origin",
            "pendrive",
            "premium",
            "react",
            "scheda",
            "secure",
            "select",
            "serie",
            "speicher",
            "tarjeta",
            "transmemo",
            "transmemory",
            "traveler",
            "ultimate",
            "ultra",
            "usb",
            "usm32gqx",
            "veloc",
            "wex",
            "xqd",
        ],
    }

    intenso_type = [
        "basic",
        "rainbow",
        "high speed",
        "speed",
        "premium",
        "alu",
        "business",
        "micro",
        "imobile",
        "cmobile",
        "mini",
        "ultra",
        "slim",
        "flash",
        "mobile",
    ]

    colors = [
        "midnight black",
        "prism white",
        "prism black",
        "prism green",
        "prism blue",
        "canary yellow",
        "flamingo pink",
        "cardinal red",
        "smoke blue",
        "deep blue",
        "coral orange",
        "black sky",
        "gold sand",
        "blue mist and peach cloud",
        "orchid gray",
        "metallic copper",
        "lavender purple",
        "ocean blue",
        "pure white",
        "alpine white",
        "copper",
        "red",
        "black",
        "blue",
        "white",
        "silver",
        "gold",
        "violet",
        "purple",
        "brown",
        "orange",
        "coral",
        "pink",
    ]

    result = []

    for row in range(data.shape[0]):
        name_info = data["name"][row]

        capacity = "0"
        mem_type = "0"
        brand = "0"
        product_type = "0"
        model = "0"
        item_code = "0"
        series = "0"
        pat_hb = "0"
        hybrid_ns = "0"
        long_num = "0"

        hybrid = re.findall(r"(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[^\W_gGM]{5,}", name_info)
        if len(hybrid) > 0:
            hybrid_ns = " ".join(sorted(hybrid))

        hybrid = re.findall(r"[0-9]{4,}", name_info)
        if len(hybrid) > 0:
            long_num = " ".join(sorted(hybrid))

        if (
            name_info
            == "pny clé usb 2.0 attaché 4 standard 8 go - noir,pny,fd8gbatt4-ef,clé usb 8 go,clé usb 8 go fantaisie,clé usb 8 go originale,clé usb 8 go pas cher,clé usb 8 go verbatim,clé usb 8 go sandisk,clé usb 8 go kingston,clé usb 8 go lexar,clé usb 8 go emtec,clé usb 8 go légo,légo,clef usb 8 go,clef usb 8 go fantaisie,clef usb 8 go originale,clef usb 8 go sandisk,clef usb 8 go verbatim,clef usb 8 go kingston,clef usb 8 go emtec,clé usb 3.0,clé usb 3.0 8 go,clef usb 3.0 8 go,clef usb 8 go 3.0,clé usb 8 go 3.0,clé usb 2.0,clé usb 2.0 8 go,cle usb,cle usb utilisation,clè usb,clès usb,cle usb windows,cléf usb,cles usb 8go,clefs usb,clé usb 8go,clé usb sandisk,clée usb,clé usb pas chère,clé usb emetec,clé usb 8 go,clé usb 8gb,usb,usb 2.0,clé usb corsair,usb key,cle usb pas cheres,clé usb pny,clé usb 8 giga,clé usb transcend,clé usb retractable,memory stick,stockage usb,cléusb,cle usb 8 g,sandisk extreme,clef usb pas cher,clé usb 8g,usb clé,meilleur clé usb,sandisk ultra,clé usb 8 go pas cher,emtec clé usb,clé usb windows 7,sandisk cruzer blade,usb flash,mémoire flash usb,clé usb 2.0,clé usb go,flash disk,sandisk cruzer fit,verbatim clé usb,meilleure clé usb,clé usb intégral,clé usb stockage,clé usb ultra rapide,clé usb 8,stockage clé usb,clé pc,pny clé usb,usb 8go,usb clef,pny usb,clé usb 8gb,clé usb emtec 8go,clé usb toshiba,clé usb 8 go,clé usb 8 go,clé usb flash drive,usb sandisk,sandisk usb,sandisk clé usb,clé de 8,clef usb 8go,clé usb petite capacité,usb 8 go"
        ):
            name_info = "pny attaché 4"

        if "intenso 3534490" in name_info:
            name_info = "memoria usb - intenso -64gb usb 3.0 (3.1 gen 1) tipo a plata"

        if name_info == "pam??ová karta kingston sdhc 16gb uhs-i u1 (90r/45w)":
            name_info = "Kingston Technology SDA10/16GB 16GB UHS-I Ultimate Flash Card".lower()

        size_model = re.findall(r"[0-9]{1,4}[ ]*[gt][bo]", name_info)
        if len(size_model) > 0:
            capacity = str(size_model[0]).replace(" ", "").replace("b", "").replace("o", "")

        mem_model = re.search(r"ssd", name_info)
        if mem_model is None:
            mem_model = re.search(r"micro[- ]?sd[hx]?c?", name_info)
        if mem_model is None:
            mem_model = re.search(r"usb", name_info)
        if mem_model is None:
            mem_model = re.search(r"sd[hx]c", name_info)
        if mem_model is None:
            mem_model = re.search(r"sd(?!cz)", name_info)
        if mem_model is None:
            mem_model = re.search(r"secure digital", name_info)
        if mem_model is None:
            mem_model = re.search(r"xqd", name_info)
        if mem_model is None:
            mem_model = re.search(r"ljd", name_info)
        if mem_model is not None:
            mem_type = mem_model.group()
            if mem_type not in ("ssd", "usb", "xqd"):
                if "micro" in mem_type:
                    mem_type = "microsd"
                elif "ljd" in mem_type:
                    mem_type = "usb"
                else:
                    mem_type = "sd"
        if "adapt" in name_info and mem_type == "0":
            mem_type = "microsd"

        pattern_hb = re.search(r"\w+-\w+", name_info)
        if pattern_hb is not None:
            pat_hb = pattern_hb.group()

        name_split_list = sorted(name_info.split())
        brand_list = []
        series_list = []
        for b in brands:
            if b in name_info:
                brand_list.append(b)
                found_series = False
                for name_debris in name_split_list:
                    for sn in families[b]:
                        if sn in name_debris:
                            series_list.append(sn)
                            found_series = True
                            break
                    if found_series:
                        break
        if len(brand_list) > 0:
            brand = "".join(sorted(brand_list))
        if len(series_list) > 0:
            series = "".join(sorted(series_list))
        if brand == "0":
            if "tos" in name_info:
                brand = "toshiba"
            # elif 'tesco' in name_info:
            #     brand = 'tesco'
            elif "cruizer glide" in name_info:
                brand = "sandisk"
            elif "hyperx" in name_info and capacity == "512g" and mem_type == "usb":
                brand = "kingston"
            elif pat_hb == "uhs-i" and mem_type == "microsd" and capacity == "8g" and "adapter" in name_info:
                brand = "kingston"
            elif "g2" in name_info and mem_type == "usb" and capacity == "16g":
                brand = "kingston"
            found_series = False
            for name_debris in name_split_list:
                for sn in families[brand]:
                    if sn in name_debris:
                        series = sn
                        found_series = True
                        break
                if found_series:
                    break

        item_code_model = re.search(r"\((mk)?[0-9]{6,10}\)", name_info)
        if item_code_model is not None:
            item_code = item_code_model.group()[1:-1]

        if brand == "intenso":
            model_model = re.search(r"[0-9]{7}", name_info)
            if model_model is not None:
                model = model_model.group()[:]
            type_model = re.search(r"(high\s)?[a-z]+\s(?=line)", name_info)
            if type_model is not None:
                product_type = type_model.group()[:].replace(" ", "")
                mem_type = "usb"
            else:
                for t in intenso_type:
                    if t in name_info:
                        product_type = t.replace(" ", "")
                        break
            if "tipo a plata" in name_info:
                product_type = "premium"
                series = "line"
        elif brand == "lexar":
            type_model = re.search(r"((jd)|[\s])[a-wy-z][0-9]{2}[a-z]?", name_info)
            if type_model is None:
                type_model = re.search(r"[\s][0-9]+x(?![a-z0-9])", name_info)
            if type_model is None:
                type_model = re.search(r"(([\s][x])|(beu))[0-9]+", name_info)
            if type_model is not None:
                product_type = (
                    type_model.group()
                    .strip()
                    .replace("x", "")
                    .replace("l", "")
                    .replace("j", "")
                    .replace("d", "")
                    .replace("b", "")
                    .replace("e", "")
                    .replace("u", "")
                )
            if mem_type == "0" and "drive" in name_info:
                mem_type = "usb"
            if "lexar 8gb jumpdrive v10 8gb usb 2.0 tipo-a blu unità flash usb" in name_info:
                product_type = "c20c"
            # if 'tarjeta' in name_info:
            #     model = 'tarjeta'
            # elif 'carte' in name_info:
            #     model = 'carte'
        elif brand == "sony":
            if mem_type == "0":
                if "ux" in name_info or "uy" in name_info or "sr" in name_info:
                    mem_type = "microsd"
                elif "uf" in name_info:
                    mem_type = "sd"
                elif "usm" in name_info or capacity == "1tb":
                    mem_type = "usb"
            type_model = re.search(r"((sf)|(usm))[-]?[0-9a-z]{1,6}", name_info)
            if type_model is not None:
                product_type = type_model.group().replace("-", "").replace("g", "")
                for c in range(ord("0"), ord("9")):
                    product_type = product_type.replace(chr(c), "")
                if "sf" in product_type and mem_type == "0":
                    mem_type = "sd"
                product_type = product_type.replace("sf", "").replace("usm", "")
            elif mem_type in ("sd", "usb"):
                if "machqx" in name_info:
                    product_type = "qx"
                elif "type-c" in name_info or "type c" in name_info:
                    product_type = "ca"
                type_model = re.search(r"(serie[s]?[\s-]?[a-z]{1,2}[\s])|([\s][a-z]{1,2}[\-]?serie[s]?)", name_info)
                if type_model is not None:
                    product_type = type_model.group().replace(" ", "").replace("-", "").replace("g", "")
                    product_type = product_type.replace("series", "").replace("serie", "")
        elif brand == "sandisk":
            item_code_model = re.search(r"\d+x", name_info)
            if item_code_model is not None:
                item_code = item_code_model.group()
            if series == "cruizer":
                series = "cruzer"
            model_model = re.search(r"ext.*(\s)?((plus)|(pro)|\+)", name_info)
            if model_model is not None:
                model = "ext+"
            else:
                model_model = re.search(r"ext(reme)?", name_info)
                if model_model is not None:
                    model = "ext"
                else:
                    model_model = re.search(r"fit", name_info)
                    if model_model is None:
                        model_model = re.search(r"glide", name_info)
                    if model_model is None:
                        model_model = re.search(r"blade", name_info)
                    if model_model is None:
                        model_model = re.search(r"cruzer", name_info)
                    if model_model is not None:
                        model = model_model.group()
                    else:
                        model_model = re.search(r"ultra(\s)?((plus)|(pro)|\+|(performance)|(android))", name_info)
                        if model_model is None:
                            model_model = re.search(
                                r"sandisk 8gb ultra sdhc memory card, class 10, read speed up to 80 mb/s \+ sd adapter",
                                name_info,
                            )
                        if model_model is None:
                            model_model = re.search(r"sandisk sdhc [0-9]+gb 80mb/s cl10\\n", name_info)
                        if model_model is not None:
                            model = "ultra+"
                        else:
                            model_model = re.search(r"ultra", name_info)
                            if model_model is not None:
                                model = "ultra"
                            else:
                                model_model = re.search(r"dual", name_info)
                                if model_model is None:
                                    model_model = re.search(r"double connect.*", name_info)
                                if model_model is not None:
                                    model = "dual"
            if "accessoires montres" in name_info:
                mem_type = "microsd"
                model = "ultra+"
                if "128 go" in name_info or capacity == "0":
                    capacity = "32g"
            if "achat mémoire" in name_info and mem_type == "sd" and capacity == "32g":
                mem_type = "microsd"
            if "adapt" in name_info and mem_type == "0":
                mem_type = "microsd"
            if mem_type == "0":
                if "drive" in name_info:
                    mem_type = "usb"
                elif "cruzer" in name_info:
                    mem_type = "usb"
                elif model in ("glide", "fit"):
                    mem_type = "usb"
            if mem_type == "sd":
                if "msd" in name_info:
                    mem_type = "msd"
                elif "sdhc" in name_info:
                    mem_type = "sdhc"
                elif "sdxc" in name_info:
                    mem_type = "sdxc"
            if "sandisk - " + capacity + " extreme en fnac.es" in name_info:
                mem_type = "usb"
            if model == "dual" and capacity == "0":
                capacity = "64g"
            if "otg" in name_info and product_type == "0":
                product_type = "otg"
        elif brand == "pny":
            type_model = re.search(r"att.*?[3-4]", name_info)
            if type_model is not None:
                product_type = type_model.group().replace(" ", "").replace("-", "")
                product_type = "att" + list(filter(lambda ch: ch in "0123456789", product_type))[0]
                if mem_type == "0":
                    mem_type = "usb"
            if name_info == "pny attaché 4":
                capacity = "8g"
        elif brand == "kingston":
            if mem_type == "0":
                if "savage" in name_info or "hx" in name_info or "hyperx" in name_info:
                    mem_type = "usb"
                elif "ultimate" in name_info:
                    mem_type = "sd"
            model_model = re.search(r"(dt[i1]0?1?)|(data[ ]?t?travel?ler)", name_info)
            if model_model is not None:
                model = "data traveler"
                type_model = re.search(r"(g[234])|(gen[ ]?[234])", name_info)
                if type_model is not None:
                    product_type = type_model.group()[-1:].replace("g", "").replace("gen", "")
            else:
                type_model = re.search(r"[\s]((g[234])|(gen[ ]?[234]))[\s]", name_info)
                if type_model is not None:
                    product_type = type_model.group().strip()[-1:].replace("g", "").replace("gen", "")
                    model = "data traveler"
            if model == "data traveler" and mem_type == "0":
                mem_type = "usb"
            if "ultimate" in name_info:
                series = "ultimate"
            if product_type == "0":
                type_model = re.search(r"\w+-\w+", name_info)
                if type_model is not None:
                    product_type = type_model.group()
                    if product_type == "flash-speicherkarte" and "uhs" in name_info:
                        product_type = "uhs-i"
                    elif product_type == "high-speed":
                        product_type = "uhs-i"
            if model == "0":
                model_model = re.search(r"[ck]lasse?\s?\d+\s", name_info)
                if model_model is not None:
                    model = "class" + re.search(r"\d+", model_model.group()).group()
                else:
                    model_model = re.search(r"[uvw]+\d+[^gm)]", name_info)
                    if model_model is not None:
                        model = model_model.group()[:-1]
            if product_type == "0":
                if "flash" in name_info:
                    product_type = "flash"
                elif "plus" in name_info:
                    product_type = "plus"
            if mem_type == "0":
                if capacity == "128g" and "Speicherkarte" in name_info and "uhs" in name_info:
                    mem_type = "sd"
            if capacity == "128g" and "uhs-i" in name_info and mem_type == "sd":
                model = "class3"
            if capacity == "16g" and mem_type == "sd" and product_type == "uhs-i" and model == "class10":
                name_info = "Kingston Carte SD Professionnelles SDA10/16GB UHS-I SDHC/SDXC Classe 10 - 16Go".lower()
            # if 'secure digital' in name_info and 'high-capacity' in name_info and '4g' in name_info:
            #     name_info = 'Kingston - carte mémoire flash - 4 Go - SDHC'.lower()
        elif brand == "samsung":
            if "lte" in name_info:
                model_model = re.search(r"[\s][a-z][0-9]{1,2}[a-z]?[\s]((plus)|\+)?", name_info)
                if model_model is None:
                    model_model = re.search(r"[\s]note[\s]?[0-9]{1,2}\+?[\s]?(ultra)?", name_info)
                if model_model is None:
                    model_model = re.search(r"prime[ ]?((plus)|\+)?", name_info)
                if model_model is not None:
                    model = model_model.group().replace(" ", "").replace("plus", "+")
                mem_type = "sim"
            elif "tv" in name_info:
                size_model = re.search(r"[0-9]{2}[- ]?inch", name_info)
                if size_model is not None:
                    capacity = size_model.group()[:2]
                mem_model = re.search(r"(hd)|(qled)|(uhd)", name_info)
                if mem_model is not None:
                    mem_type = mem_model.group()
                model_model = re.search(r"[a-z]{1,2}[0-9]{4}", name_info)
                if model_model is not None:
                    model = model_model.group()
            else:
                if mem_type == "ssd":
                    model_model = re.search(r"[\s]t[0-9][\s]", name_info)
                    if model_model is not None:
                        model = model_model.group().strip()
                else:
                    model_model = re.search(r"(pro)|(evo)", name_info)
                    if model_model is not None:
                        model = model_model.group()
                        model_model = re.search(r"(\+)|(plus)", name_info)
                        if model_model is not None:
                            model = model + model_model.group().replace("plus", "+")
                        if model == "evo+" and mem_type == "0":
                            mem_type = "microsd"
            for c in colors:
                if c in name_info:
                    product_type = c
                    break
        elif brand == "toshiba":
            model_model = re.search(r"[\s\-n][umn][0-9]{3}", name_info)
            if model_model is not None:
                model = model_model.group()[1:]
                if mem_type == "0":
                    ch = model[0]
                    if ch == "u":
                        mem_type = "usb"
                    elif ch == "n":
                        mem_type = "sd"
                    elif ch == "m":
                        mem_type = "microsd"
            if mem_type == "usb" and model == "0":
                model_model = re.search(r"ex[\s-]?ii", name_info)
                if model_model is None:
                    model_model = re.search(r"osus", name_info)
                if model_model is not None:
                    model = "ex"
            if "transmemory" in name_info:
                if mem_type == "0":
                    mem_type = "usb"
            if mem_type != "usb":
                type_model = re.search(r"exceria[ ]?((high)|(plus)|(pro))?", name_info)
                if type_model is not None:
                    product_type = type_model.group().replace(" ", "").replace("exceria", "x")
                elif capacity != "0":
                    type_model = re.search(r"x[ ]?((high)|(plus)|(pro))?" + capacity[:-2], name_info)
                    if type_model is not None:
                        product_type = type_model.group().replace(" ", "")[: -(len(capacity) - 2)]
                if product_type == "xpro" and mem_type == "0":
                    mem_type = "sd"
                if product_type == "xhigh" and mem_type == "0":
                    mem_type = "microsd"
            if mem_type == "usb" and model == "0":
                if "hayaqa" in name_info or "hayabusa" in name_info:
                    model = "u202"
            if mem_type == "sd" and model == "0":
                model_model = re.search(r"silber", name_info)
                if model_model is not None:
                    model = "n401"
            if mem_type == "sd" and model == "0" and product_type == "0":
                model_model = re.search(r"sd[hx]c uhs clas[se] 3 memor(y|(ia)) ((card)|(flash))", name_info)
                if model_model is not None:
                    product_type = "xpro"
            if mem_type == "sd" and model == "0" and product_type == "0":
                if "uhs-ii" in name_info and "carte mémoire flash" in name_info:
                    product_type = "xpro"
            if mem_type != "usb":
                speed_model = re.search(r"[1-9][0-9]{1,2}[\s]?m[bo]/s", name_info)
                if speed_model is not None:
                    speed = re.search(r"[0-9]{2,3}", speed_model.group()).group()
                    if speed == "260" or speed == "270":
                        if product_type == "0":
                            product_type = "xpro"
                    if speed == "90" and product_type == "x":
                        if model == "0":
                            model = "n302"
            if "gb uhs-i (u3 - up to 95mb/s read) flash memory card" in name_info:
                mem_type = "microsd"
                product_type = "x"
            if "toshiba pendrive usb high-speed" in name_info:
                model = "u202"
            if "en fnac.es" in name_info and "toshiba usb 3.0" in name_info and "pendrive / memoria usb" in name_info:
                model = "ex"
            # if model == 'n101':
            #     model = '0'
            if "memoria toshiba exceria microsdxc clase 10 uhs-i class 3 16gb" in name_info:
                capacity = "64g"
            if "sd-xpro32uhs2" in name_info:
                series = "exceria"
            if model == "0" and "uhs" in name_info:
                model = "uhs"
        elif brand == "transcend":
            if model == "0" and "uhs" in name_info:
                model = "uhs"

        result.append(
            [
                data["id"][row],
                brand,
                capacity,
                mem_type,
                product_type,
                model,
                item_code,
                series,
                pat_hb,
                hybrid_ns,
                long_num,
                name_info,
            ]
        )

    result = pd.DataFrame(
        result,
        columns=[
            "id",
            "brand",
            "capacity",
            "mem_type",
            "type",
            "model",
            "item_code",
            "series",
            "pat_hb",
            "hybrid",
            "long_num",
            "name",
        ],
    )
    return result
