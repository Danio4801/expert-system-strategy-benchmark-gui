




















import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from core.experiment_manager import ExperimentConfig

from core.inference import InferenceResult
from core.models import Rule


logger = logging.getLogger(__name__)


class ExperimentStorage:





















    def __init__(self, base_dir: str = "user_experiments"):






        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(f"{__name__}.ExperimentStorage")


        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"ExperimentStorage zainicjalizowany. Katalog bazowy: {self.base_dir}")

    def save_experiment(
        self,
        run_id: str,
        dataset_name: str,
        config: "ExperimentConfig",
        result: InferenceResult,
        rules: List[Rule],
        inference_engine: Optional[object] = None
    ) -> Path:

























        if not run_id:
            raise ValueError("run_id nie może być pusty")
        if not dataset_name:
            raise ValueError("dataset_name nie może być pusty")

        self.logger.info("="*70)
        self.logger.info("ZAPIS EKSPERYMENTU NA DYSK")
        self.logger.info("="*70)
        self.logger.info(f"Run ID: {run_id}")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Method: {config.generate_method.value}")


        experiment_dir = self._create_experiment_directory(run_id, dataset_name, config)


        self._save_metadata(experiment_dir, run_id, config, result, dataset_name, len(rules), inference_engine)


        self._save_rules(experiment_dir, rules)


        self._copy_logs(experiment_dir, run_id)

        self.logger.info("="*70)
        self.logger.info(f"EKSPERYMENT ZAPISANY: {experiment_dir}")
        self.logger.info("="*70)

        return experiment_dir

    def _create_experiment_directory(
        self,
        run_id: str,
        dataset_name: str,
        config: "ExperimentConfig"
    ) -> Path:















        safe_dataset_name = self._sanitize_filename(dataset_name)


        dir_name = f"{run_id}_{safe_dataset_name}_{config.generate_method.value}"
        experiment_dir = self.base_dir / dir_name


        experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"[STORAGE] Utworzono katalog: {experiment_dir}")

        return experiment_dir

    def _save_metadata(
        self,
        experiment_dir: Path,
        run_id: str,
        config: "ExperimentConfig",
        result: InferenceResult,
        dataset_name: str,
        rules_count: int,
        inference_engine: Optional[object] = None
    ):



































        config_dict = asdict(config)
        config_dict['strategy'] = config.strategy.value
        config_dict['generate_method'] = config.generate_method.value
        config_dict['inference_method'] = config.inference_method.value


        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "seed": config.seed,
            "strategy": config.strategy.value,
            "generate_method": config.generate_method.value,
            "inference_method": config.inference_method.value,
            "metrics": {
                "execution_time_ms": result.execution_time_ms,
                "rules_count": rules_count,
                "new_facts_count": len(result.new_facts),
                "iterations": result.iterations,
                "rules_evaluated": result.rules_evaluated,
                "rules_activated": result.rules_activated,
                "facts_count": result.facts_count
            },
            "config": config_dict,
            "success": result.success
        }


        if inference_engine is not None and hasattr(inference_engine, 'clusters_checked'):
            metadata["metrics"]["clusters_checked"] = inference_engine.clusters_checked
            metadata["metrics"]["clusters_skipped"] = inference_engine.clusters_skipped
            metadata["metrics"]["skip_rate"] = inference_engine.clusters_skipped / len(inference_engine.clusters) if inference_engine.clusters else 0.0

            if hasattr(inference_engine, 'centroid_evaluations'):
                metadata["metrics"]["centroid_evaluations"] = inference_engine.centroid_evaluations
                metadata["metrics"]["total_rules_in_clusters"] = sum(c.size for c in inference_engine.clusters) if inference_engine.clusters else 0


        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"[STORAGE] Zapisano metadata.json ({metadata_path.stat().st_size} bytes)")

    def _save_rules(self, experiment_dir: Path, rules: List[Rule]):












        rules_path = experiment_dir / "rules.txt"

        with open(rules_path, 'w', encoding='utf-8') as f:
            f.write(f"# Generated rules - Total: {len(rules)}\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#" + "="*68 + "\n\n")

            for rule in rules:
                f.write(f"{rule}\n")

        self.logger.info(f"[STORAGE] Saved {len(rules)} rules to rules.txt")

    def _copy_logs(self, experiment_dir: Path, run_id: str):













        logs_dir = Path("logs")


        log_files = [
            f"inference_{run_id}.log",
            f"inference_{run_id}_extended.log"
        ]

        copied_count = 0

        for log_file in log_files:
            source_path = logs_dir / log_file
            dest_path = experiment_dir / log_file

            if source_path.exists():
                try:
                    shutil.copy2(source_path, dest_path)
                    self.logger.info(f"[STORAGE] Copied log: {log_file}")
                    copied_count += 1
                except Exception as e:
                    self.logger.warning(f"[STORAGE] Failed to copy {log_file}: {e}")
            else:
                self.logger.warning(f"[STORAGE] Log file does not exist: {source_path}")

        if copied_count == 0:
            self.logger.warning("[STORAGE] No log files were copied")
        else:
            self.logger.info(f"[STORAGE] Copied {copied_count}/{len(log_files)} log files")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:










        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')


        filename = filename.strip()

        return filename

    def load_experiment_metadata(self, experiment_dir: Path) -> Optional[Dict]:











        metadata_path = experiment_dir / "metadata.json"

        if not metadata_path.exists():
            self.logger.error(f"[STORAGE] No metadata.json in {experiment_dir}")
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.logger.info(f"[STORAGE] Loaded metadata from {metadata_path}")
            return metadata
        except Exception as e:
            self.logger.error(f"[STORAGE] Error loading metadata: {e}")
            return None

    def list_experiments(self) -> List[Path]:








        if not self.base_dir.exists():
            return []


        experiments = [d for d in self.base_dir.iterdir() if d.is_dir()]


        experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        self.logger.info(f"[STORAGE] Found {len(experiments)} experiments")

        return experiments

    def load_log_file(self, experiment_dir: Path, log_type: str = "extended") -> Optional[str]:











        folder_name = experiment_dir.name


        import re
        match = re.match(r'(exp_\d{8}_\d{6})', folder_name)

        if not match:
            self.logger.error(f"[STORAGE] Cannot extract run_id from folder name: {folder_name}")
            return None

        run_id = match.group(1)


        if log_type == "extended":
            log_filename = f"inference_{run_id}_extended.log"
        else:
            log_filename = f"inference_{run_id}.log"

        log_path = experiment_dir / log_filename

        if not log_path.exists():
            self.logger.warning(f"[STORAGE] Log file does not exist: {log_path}")
            return None

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"[STORAGE] Loaded log {log_filename} ({len(content)} characters)")
            return content
        except Exception as e:
            self.logger.error(f"[STORAGE] Error loading log: {e}")
            return None

    def get_experiment_summary(self, experiment_dir: Path) -> Optional[Dict]:









        metadata = self.load_experiment_metadata(experiment_dir)

        if not metadata:
            return None



        has_logs = any(experiment_dir.glob("*.log"))

        summary = {
            'folder_path': experiment_dir,
            'folder_name': experiment_dir.name,
            'run_id': metadata.get('run_id'),
            'timestamp': metadata.get('timestamp'),
            'dataset_name': metadata.get('dataset_name'),
            'generate_method': metadata.get('generate_method'),
            'inference_method': metadata.get('inference_method'),
            'strategy': metadata.get('strategy'),
            'success': metadata.get('success'),
            'execution_time_ms': metadata.get('metrics', {}).get('execution_time_ms'),
            'rules_count': metadata.get('metrics', {}).get('rules_count'),
            'new_facts_count': metadata.get('metrics', {}).get('new_facts_count'),
            'has_logs': has_logs,

            'sync_status': 'local',
        }

        return summary
