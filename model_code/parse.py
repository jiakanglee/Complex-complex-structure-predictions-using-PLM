class OpenFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir: str,
        alignment_dir: str, 
        template_mmcif_dir: str,
        max_template_date: str,
        config: mlc.ConfigDict,
        chain_data_cache_path: Optional[str] = None,
        kalign_binary_path: str = '/usr/bin/kalign',
        max_template_hits: int = 4,
        obsolete_pdbs_file_path: Optional[str] = None,
        template_release_dates_cache_path: Optional[str] = None,
        shuffle_top_k_prefiltered: Optional[int] = None,
        treat_pdb_as_distillation: bool = True,
        filter_path: Optional[str] = None,
        mode: str = "train", 
        alignment_index: Optional[Any] = None,
        _output_raw: bool = False,
        _structure_index: Optional[Any] = None,
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                chain_data_cache_path:
                    Path to cache of data_dir generated by
                    scripts/generate_chain_data_cache.py
                kalign_binary_path:
                    Path to kalign binary.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(OpenFoldSingleDataset, self).__init__()
        self.data_dir = data_dir

        self.chain_data_cache = None
        if chain_data_cache_path is not None:
            with open(chain_data_cache_path, "r") as fp:
                self.chain_data_cache = json.load(fp)
            assert isinstance(self.chain_data_cache, dict)

        self.alignment_dir = alignment_dir
        self.config = config
        self.treat_pdb_as_distillation = treat_pdb_as_distillation
        self.mode = mode
        self.alignment_index = alignment_index
        self._output_raw = _output_raw
        self._structure_index = _structure_index

        self.supported_exts = [".cif", ".core", ".pdb"]

        valid_modes = ["train", "eval", "predict"]
        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')

        if(template_release_dates_cache_path is None):
            logging.warning(
                "Template release dates cache does not exist. Remember to run "
                "scripts/generate_mmcif_cache.py before running OpenFold"
            )

        if(alignment_index is not None):
            self._chain_ids = list(alignment_index.keys())
        else:
            self._chain_ids = list(os.listdir(alignment_dir))

        if(filter_path is not None):
            with open(filter_path, "r") as f:
                chains_to_include = set([l.strip() for l in f.readlines()])

            self._chain_ids = [
                c for c in self._chain_ids if c in chains_to_include
            ]

        if self.chain_data_cache is not None:
            # Filter to include only chains where we have structure data
            # (entries in chain_data_cache)
            original_chain_ids = self._chain_ids
            self._chain_ids = [
                c for c in self._chain_ids if c in self.chain_data_cache
            ]
            if len(self._chain_ids) < len(original_chain_ids):
                missing = [
                    c for c in original_chain_ids
                    if c not in self.chain_data_cache
                ]
                max_to_print = 10
                missing_examples = ", ".join(missing[:max_to_print])
                if len(missing) > max_to_print:
                    missing_examples += ", ..."
                logging.warning(
                    "Removing %d alignment entries (%s) with no corresponding "
                    "entries in chain_data_cache (%s).",
                    len(missing),
                    missing_examples,
                    chain_data_cache_path)
       
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_template_hits,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=template_release_dates_cache_path,
            obsolete_pdbs_path=obsolete_pdbs_file_path,
            _shuffle_top_k_prefiltered=shuffle_top_k_prefiltered,
        )

        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        if(not self._output_raw):
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config) 

    def _parse_mmcif(self, path, file_id, chain_id, alignment_dir, alignment_index):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        # Crash if an error is encountered. Any parsing errors should have
        # been dealt with at the alignment stage.
        if(mmcif_object.mmcif_object is None):
            raise list(mmcif_object.errors.values())[0]

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            alignment_dir=alignment_dir,
            chain_id=chain_id,
            alignment_index=alignment_index,
            seqemb_mode=self.config.seqemb_mode.enabled
        )

        return data

    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]

    def __getitem__(self, idx):
        name = self.idx_to_chain_id(idx)
        alignment_dir = os.path.join(self.alignment_dir, name)

        alignment_index = None
        if(self.alignment_index is not None):
            alignment_dir = self.alignment_dir
            alignment_index = self.alignment_index[name]

        if(self.mode == 'train' or self.mode == 'eval'):
            spl = name.rsplit('_', 1)
            if(len(spl) == 2):
                file_id, chain_id = spl
            else:
                file_id, = spl
                chain_id = None

            path = os.path.join(self.data_dir, file_id)
            structure_index_entry = None
            if(self._structure_index is not None):
                structure_index_entry = self._structure_index[name]
                assert(len(structure_index_entry["files"]) == 1)
                filename, _, _ = structure_index_entry["files"][0]
                ext = os.path.splitext(filename)[1]
            else:
                ext = None
                for e in self.supported_exts:
                    if(os.path.exists(path + e)):
                        ext = e
                        break

                if(ext is None):
                    raise ValueError("Invalid file type")

            path += ext
            if(ext == ".cif"):
                data = self._parse_mmcif(
                    path, file_id, chain_id, alignment_dir, alignment_index,
                )
            elif(ext == ".core"):
                data = self.data_pipeline.process_core(
                    path, alignment_dir, alignment_index,
                    seqemb_mode=self.config.seqemb_mode.enabled,
                )
            elif(ext == ".pdb"):
                structure_index = None
                if(self._structure_index is not None):
                    structure_index = self._structure_index[name]
                data = self.data_pipeline.process_pdb(
                    pdb_path=path,
                    alignment_dir=alignment_dir,
                    is_distillation=self.treat_pdb_as_distillation,
                    chain_id=chain_id,
                    alignment_index=alignment_index,
                    _structure_index=structure_index,
                    seqemb_mode=self.config.seqemb_mode.enabled,
                )
            else:
               raise ValueError("Extension branch missing") 
        else:
            path = os.path.join(name, name + ".fasta")
            data = self.data_pipeline.process_fasta(
                fasta_path=path,
                alignment_dir=alignment_dir,
                alignment_index=alignment_index,
                seqemb_mode=self.config.seqemb_mode.enabled,
            )

        if(self._output_raw):
            return data

        feats = self.feature_pipeline.process_features(
            data, self.mode 
        )

        feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["aatype"].shape[-1])],
            dtype=torch.int64,
            device=feats["aatype"].device)

        return feats

    def __len__(self):
        return len(self._chain_ids) 