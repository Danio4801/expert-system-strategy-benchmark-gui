




import flet as ft
from typing import List, Optional, Set
import os
import json
import csv
import random
import pandas as pd
import sys
import time
import threading
from datetime import datetime
from firebase_service import FirebaseService
from csv_loader import load_csv, CSVLoadError, print_metadata
from translations import lang
from utils.app_state import AppStateManager



if hasattr(sys, '_MEIPASS'):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.append('..')
from core.models import Fact

from core.experiment_manager import (
    ExperimentConfig,
    ExperimentRunner,
    InferenceStrategy,
    RuleGenerationMethod,
    InferenceMethod
)

from core.inference import ForwardChaining, GreedyForwardChaining, BackwardChaining
from core.strategies import RandomStrategy, FirstStrategy, SpecificityStrategy, RecencyStrategy
from core.models import KnowledgeBase, Rule

from preprocessing.bin_suggester import BinSuggester, BinSuggestion



def resource_path(relative_path):












    try:

        base_path = sys._MEIPASS
    except Exception:

        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



class AppSettings:


    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not AppSettings._initialized:

            self.detailed_logs = True
            self.auto_export = True


            self.keep_logged_in = True

            AppSettings._initialized = True


app_settings = AppSettings()


class AppColors:

    PRIMARY = "#6366F1"
    PRIMARY_DARK = "#4F46E5"
    PRIMARY_LIGHT = "#818CF8"
    SECONDARY = "#10B981"
    WARNING = "#F59E0B"
    ERROR = "#EF4444"

    BG_DARK = "#0F172A"
    BG_CARD = "#1E293B"
    BG_ELEVATED = "#334155"

    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#E2E8F0"
    TEXT_MUTED = "#CBD5E1"

    BORDER = "#475569"


    SUCCESS_BG = "#064E3B"
    SECONDARY_BG = "#064E3B"
    ERROR_BG = "#7F1D1D"
    WARNING_BG = "#78350F"

class AppFonts:

    HEADING = "Poppins"
    BODY = "Source Sans Pro"
    MONO = "JetBrains Mono"


def create_card(content: ft.Control, padding: int = 20) -> ft.Container:

    return ft.Container(
        content=content,
        padding=padding,
        border_radius=12,
        bgcolor=AppColors.BG_CARD,
        border=ft.border.all(1, AppColors.BORDER),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color=ft.colors.with_opacity(0.3, "#000000"),
            offset=ft.Offset(0, 4)
        )
    )

def create_stat_card(title: str, value: str, icon: str, color: str) -> ft.Container:

    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(icon, color=color, size=24),
                ft.Text(title, size=12, color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
            ], spacing=8),
            ft.Text(value, size=28, color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
        ], spacing=8),
        padding=20,
        border_radius=12,
        bgcolor=AppColors.BG_CARD,
        border=ft.border.all(1, AppColors.BORDER),
        expand=True,
    )

def create_section_header(title: str, subtitle: str = None) -> ft.Column:

    children = [
        ft.Text(title, size=20, color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
    ]
    if subtitle:
        children.append(
            ft.Text(subtitle, size=13, color=AppColors.TEXT_SECONDARY)
        )
    return ft.Column(children, spacing=4)

def create_chip(text: str, color: str = AppColors.PRIMARY) -> ft.Container:

    return ft.Container(
        content=ft.Text(text, size=11, color=color, weight=ft.FontWeight.W_600),
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        border_radius=20,
        bgcolor=ft.colors.with_opacity(0.15, color),
    )

def create_placeholder_chart(title: str, chart_type: str, height: int = 200) -> ft.Container:
    icons = {
        "bar": ft.icons.BAR_CHART,
        "pie": ft.icons.PIE_CHART,
        "line": ft.icons.SHOW_CHART,
        "radar": ft.icons.RADAR,
        "scatter": ft.icons.SCATTER_PLOT,
    }
    return ft.Container(
        content=ft.Column([
            ft.Icon(icons.get(chart_type, ft.icons.INSERT_CHART),
                   size=48, color=AppColors.TEXT_MUTED),
            ft.Text(f" {title}", size=14, color=AppColors.TEXT_SECONDARY,
                   weight=ft.FontWeight.W_500),
            ft.Text(f"{lang.t('chart_placeholder')} {chart_type}", size=12,
                   color=AppColors.TEXT_MUTED, italic=True),
        ], alignment=ft.MainAxisAlignment.CENTER,
           horizontal_alignment=ft.CrossAxisAlignment.CENTER,
           spacing=8),
        height=height,
        border_radius=8,
        bgcolor=AppColors.BG_ELEVATED,
        border=ft.border.all(2, ft.colors.with_opacity(0.3, AppColors.BORDER)),
        alignment=ft.alignment.center,
    )


class Sidebar(ft.UserControl):
    def __init__(self, on_navigate):
        super().__init__()
        self.on_navigate = on_navigate
        self.selected_index = -1
        self.firebase = FirebaseService()
        self._update_menu_items()

    def _update_menu_items(self):

        self.menu_items = [
            (lang.t('nav_new_experiment'), ft.icons.SCIENCE_ROUNDED, 0),
            (lang.t('nav_knowledge_base'), ft.icons.STORAGE_ROUNDED, 1),
            (lang.t('nav_results'), ft.icons.ANALYTICS_ROUNDED, 2),
            (lang.t('nav_settings'), ft.icons.SETTINGS_ROUNDED, 3),
        ]

    def _get_greeting(self):

        import datetime
        hour = datetime.datetime.now().hour

        if 5 <= hour < 12:
            return lang.t('greeting_morning')
        elif 12 <= hour < 18:
            return lang.t('greeting_afternoon')
        elif 18 <= hour < 22:
            return lang.t('greeting_evening')
        else:
            return lang.t('greeting_night')

    def _get_user_initial(self):

        if self.firebase.is_logged_in() and self.firebase.current_user:

            username = self.firebase.current_user.get('username', 'Guest')

            return username[0].upper() if username else 'G'
        else:
            return 'G'

    def _get_user_name(self):

        if self.firebase.is_logged_in() and self.firebase.current_user:
            return self.firebase.current_user.get('username', lang.t('user_guest'))
        else:
            return lang.t('user_guest')

    def build(self):

        self.menu_container = ft.Column(spacing=4)
        self._update_menu()


        self.user_avatar = ft.CircleAvatar(
            content=ft.Text(self._get_user_initial(), size=14, weight=ft.FontWeight.BOLD),
            bgcolor=AppColors.PRIMARY,
            radius=18,
        )

        self.user_name_text = ft.Text(
            self._get_user_name(),
            size=13,
            color=AppColors.TEXT_PRIMARY,
            weight=ft.FontWeight.W_500
        )

        self.user_greeting_text = ft.Text(
            self._get_greeting(),
            size=11,
            color=AppColors.TEXT_MUTED
        )

        return ft.Container(
            content=ft.Column([

                ft.Container(
                    content=ft.Column([
                        ft.Text(lang.t('app_title'), size=18, weight=ft.FontWeight.BOLD,
                               color=AppColors.TEXT_PRIMARY),
                        ft.Text(lang.t('app_subtitle'), size=18, weight=ft.FontWeight.W_300,
                               color=AppColors.PRIMARY_LIGHT),
                    ], spacing=0),
                    padding=ft.padding.only(left=16, top=20, bottom=30),
                ),


                self.menu_container,


                ft.Container(expand=True),


                ft.Container(
                    content=ft.Row([
                        self.user_avatar,
                        ft.Column([
                            self.user_name_text,
                            self.user_greeting_text,
                        ], spacing=2),
                    ], spacing=10),
                    padding=16,
                    border=ft.border.only(top=ft.BorderSide(1, AppColors.BORDER)),
                ),
            ]),
            width=240,
            bgcolor=AppColors.BG_CARD,
            border=ft.border.only(right=ft.BorderSide(1, AppColors.BORDER)),
        )

    def _update_menu(self):

        def create_menu_item(text: str, icon, index: int):
            is_selected = index == self.selected_index
            return ft.Container(
                content=ft.Row([
                    ft.Icon(icon,
                           color=AppColors.PRIMARY if is_selected else AppColors.TEXT_SECONDARY,
                           size=20),
                    ft.Text(text,
                           size=14,
                           color=AppColors.TEXT_PRIMARY if is_selected else AppColors.TEXT_SECONDARY,
                           weight=ft.FontWeight.W_600 if is_selected else ft.FontWeight.W_400),
                ], spacing=12),
                padding=ft.padding.symmetric(horizontal=16, vertical=12),
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if is_selected else None,
                on_click=lambda e, idx=index: self._on_click(idx),
                ink=True,
            )

        self.menu_container.controls = [
            create_menu_item(text, icon, idx)
            for text, icon, idx in self.menu_items
        ]

    def _on_click(self, index: int):
        self.selected_index = index
        self._update_menu()
        self.update()
        self.on_navigate(index)

    def refresh_user_info(self):

        if hasattr(self, 'user_avatar') and hasattr(self, 'user_name_text') and hasattr(self, 'user_greeting_text'):

            self.user_avatar.content.value = self._get_user_initial()


            self.user_name_text.value = self._get_user_name()


            self.user_greeting_text.value = self._get_greeting()


            self.update()



class NewExperimentView(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.current_step = 0

        self.selected_dataset = None

        self.selected_discretization = None
        self.selected_imputation = None


        self.imputation_completed = False
        self.discretization_completed = False

        self.imputation_numeric_method = "mean"
        self.imputation_categorical_method = "mode"
        self.imputation_report = None
        self.bins_choice = None
        self.bin_suggestion: Optional[BinSuggestion] = None
        self.selected_strategy = None
        self.selected_strategies = set()
        self.n_bins = 5


        self.available_columns = ["Kolumna 1", "Kolumna 2", "Kolumna 3", "Kolumna 4", "Kolumna 5"]
        self.selected_columns = set(self.available_columns)
        self.disc_details_initialized = False


        self.initial_facts_percent = "10, 25, 50"
        self.repetitions = "50"
        self.facts_validation_error = None
        self.repetitions_validation_error = None


        self.random_seed = "26"
        self.skip_validation = True
        self.seed_validation_error = None


        self.firebase = FirebaseService()
        self.user_files = []


        self.local_files_path = os.path.join(
            os.path.dirname(__file__),
            'local_files.json'
        )
        self.local_files = self._load_local_files()


        self.loaded_file_path = None
        self.loaded_df = None
        self.loaded_metadata = None
        self.discretized_df = None
        self.file_status_message = None
        self.file_status_container = None


        self.selected_rule_method = None


        self.tree_max_depth = 3
        self.tree_min_samples_leaf = 5


        self.rf_min_depth = 2
        self.rf_max_depth = 12
        self.rf_min_samples_leaf = 5
        self.rf_n_estimators = 100

        self.generated_rules = None


        self.selected_algorithm = "Forward Chaining"
        self.use_clustering = False
        self.n_clusters = 10
        self.centroid_method = "specialized"
        self.centroid_threshold = 0.3
        self.centroid_match_threshold = 0.0
        self.use_greedy = False
        self.use_forward_goal = False
        self.forward_goal_attr = None
        self.forward_goal_value = None
        self.forward_goal_any_value = False
        self.backward_goal_attr = None
        self.backward_goal_value = None
        self.backward_goal_any_value = False
        self.inference_config = {}


        self.inference_results_container = ft.Container()


        self.benchmark_progress_bar = None
        self.benchmark_status_text = None


        self.preload_file_path = None


        self.csv_column_separator = ','
        self.csv_decimal_separator = '.'
        self.csv_has_header = True
        self.csv_encoding = 'utf-8'
        self.csv_preview_lines = []
        self.csv_detected_columns = []
        self.csv_decision_column = None


        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)
        self.page = None
        
    def _get_steps(self):








        base_steps = [
            "Konfiguracja eksperymentu",
            lang.t('new_exp_step_data'),
            lang.t('new_exp_step_csv_config'),
            lang.t('new_exp_step_imputation'),
            lang.t('new_exp_step_discretization'),
        ]


        if self.bins_choice == "manual":
            base_steps.append(lang.t('new_exp_step_disc_details'))

        base_steps.extend([
            lang.t('new_exp_step_rule_generation'),
            lang.t('new_exp_step_algorithm'),
            lang.t('new_exp_step_strategy'),
            lang.t('new_exp_step_run')
        ])

        return base_steps

    def build(self):

        self.steps = self._get_steps()


        self.stepper_container = ft.Row(spacing=8)


        self.stepper_row = ft.Row(
            controls=[self.stepper_container],
            scroll=ft.ScrollMode.HIDDEN,
        )

        self._update_stepper()


        self.content_container = ft.Container(padding=30)
        self._update_content()


        self.back_button = ft.OutlinedButton(
            lang.t('back'),
            icon=ft.icons.ARROW_BACK_ROUNDED,
            visible=False,
            style=ft.ButtonStyle(
                color=AppColors.TEXT_PRIMARY,
                side=ft.BorderSide(1, AppColors.BORDER),
            ),
            on_click=self._prev_step,
        )


        self.restart_button = ft.TextButton(
            "Rozpocznij od nowa",
            icon=ft.icons.RESTART_ALT_ROUNDED,
            visible=False,
            style=ft.ButtonStyle(
                color=AppColors.TEXT_MUTED,
            ),
            on_click=self._restart_experiment,
        )

        self.next_button = ft.ElevatedButton(
            lang.t('next'),
            icon=ft.icons.ARROW_FORWARD_ROUNDED,
            style=ft.ButtonStyle(
                bgcolor=AppColors.PRIMARY,
                color=AppColors.TEXT_PRIMARY,
                padding=ft.padding.symmetric(horizontal=24, vertical=12),
            ),
            on_click=self._next_step,
        )


        self._update_navigation_buttons()


        if self.preload_file_path:
            self._auto_load_file(self.preload_file_path)

        return ft.Column([
            self.file_picker,

            ft.Text(lang.t('new_exp_title'), size=28, weight=ft.FontWeight.BOLD,
                   color=AppColors.TEXT_PRIMARY),
            ft.Text(lang.t('new_exp_subtitle'), size=14,
                   color=AppColors.TEXT_SECONDARY),
            
            ft.Container(height=30),


            create_card(self.stepper_row, padding=24),

            ft.Container(height=20),


            self.content_container,

            ft.Container(height=20),


            ft.Row([
                self.back_button,
                self.restart_button,
                ft.Container(expand=True),
                self.next_button,
            ]),
        ], scroll=ft.ScrollMode.AUTO)

    def did_mount(self):


        self._scroll_stepper_to_step(self.current_step)

    def _scroll_stepper_to_step(self, step_index: int):



        if not (hasattr(self, 'stepper_row') and self.stepper_row):
            return
        try:

            if not getattr(self.stepper_row, '_Control__page', None):
                return

            estimated_step_width = 160

            target_offset = max(0, (step_index - 1) * estimated_step_width)
            self.stepper_row.scroll_to(offset=target_offset, duration=300)
        except AssertionError:

            pass

    def _show_snackbar(self, message: str, success: bool = True):







        if self.page:
            snackbar = ft.SnackBar(
                content=ft.Text(message, color=ft.colors.WHITE),
                bgcolor=AppColors.SECONDARY if success else AppColors.ERROR,
                duration=3000,
            )
            self.page.snack_bar = snackbar
            snackbar.open = True
            self.page.update()

    def _reset_experiment_state(self):
















        print(f"[RESET] Centralny reset stanu eksperymentu...")


        self.discretized_df = None
        self.generated_rules = None


        self.bin_suggestion = None
        self.imputation_report = None


        self.inference_config = {}


        self.disc_details_initialized = False


        self.selected_imputation = None
        self.selected_discretization = None
        self.selected_strategy = None
        self.selected_strategies = set()
        self.bins_choice = None


        self.imputation_completed = False
        self.discretization_completed = False


        self.loaded_df = None
        self.csv_column_separator = ","
        self.csv_decision_column = None


        if hasattr(self, 'inference_results_container') and self.inference_results_container:
            self.inference_results_container.content = None


        self.current_step = 1

        print("[RESET] Stan wyczyszczony - gotowy na nowy eksperyment")

    def _update_stepper(self):

        def create_step(index: int, title: str):
            is_active = index == self.current_step
            is_completed = index < self.current_step

            if is_completed:
                circle_content = ft.Icon(ft.icons.CHECK, size=16, color=AppColors.TEXT_PRIMARY)
                circle_bg = AppColors.SECONDARY
            elif is_active:
                circle_content = ft.Text(str(index + 1), size=12,
                                        color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.BOLD)
                circle_bg = AppColors.PRIMARY
            else:
                circle_content = ft.Text(str(index + 1), size=12,
                                        color=AppColors.TEXT_MUTED, weight=ft.FontWeight.W_500)
                circle_bg = AppColors.BG_ELEVATED

            return ft.Row([
                ft.Container(
                    content=circle_content,
                    width=32,
                    height=32,
                    border_radius=16,
                    bgcolor=circle_bg,
                    alignment=ft.alignment.center,
                    on_click=lambda e, idx=index: self._jump_to_step(idx),
                    ink=True,
                ),
                ft.Container(
                    content=ft.Text(title, size=13,
                           color=AppColors.TEXT_PRIMARY if is_active else AppColors.TEXT_MUTED,
                           weight=ft.FontWeight.W_600 if is_active else ft.FontWeight.W_400),
                    on_click=lambda e, idx=index: self._jump_to_step(idx),
                    ink=True,
                ),
                ft.Container(
                    width=60,
                    height=2,
                    bgcolor=AppColors.SECONDARY if is_completed else AppColors.BG_ELEVATED,
                ) if index < len(self.steps) - 1 else ft.Container(),
            ], spacing=12)

        self.stepper_container.controls = [create_step(i, s) for i, s in enumerate(self.steps)]


        self._scroll_stepper_to_step(self.current_step)

    def _update_content(self):





        step_map = {
            "Konfiguracja eksperymentu": self._build_step_config,
            lang.t('new_exp_step_data'): self._build_step_data,
            lang.t('new_exp_step_csv_config'): self._build_step_csv_config,
            lang.t('new_exp_step_imputation'): self._build_step_imputation,
            lang.t('new_exp_step_discretization'): self._build_step_discretization,
            lang.t('new_exp_step_disc_details'): self._build_step_disc_details,
            lang.t('new_exp_step_rule_generation'): self._build_step_rule_generation,
            lang.t('new_exp_step_algorithm'): self._build_step_algorithm,
            lang.t('new_exp_step_strategy'): self._build_step_strategy,
            lang.t('new_exp_step_run'): self._build_step_run,
        }


        current_steps = self._get_steps()
        current_step_name = current_steps[self.current_step]


        content = step_map[current_step_name]()

        self.content_container.content = ft.Column([content])
        self.content_container.padding = 30
        self.content_container.border_radius = 12
        self.content_container.bgcolor = AppColors.BG_CARD
        self.content_container.border = ft.border.all(1, AppColors.BORDER)
        self.content_container.shadow = ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color=ft.colors.with_opacity(0.3, "#000000"),
            offset=ft.Offset(0, 4)
        )

    def _update_navigation_buttons(self):





        self.back_button.visible = self.current_step > 0


        self.restart_button.visible = self.current_step > 0



        current_step_name = self.steps[self.current_step]
        if current_step_name == lang.t('new_exp_step_csv_config'):
            self.next_button.visible = False
        else:
            self.next_button.visible = True

        if self.current_step < len(self.steps) - 1:
            self.next_button.text = lang.t('next')
            self.next_button.icon = ft.icons.ARROW_FORWARD_ROUNDED
        else:

            self.next_button.text = "Uruchom eksperyment"
            self.next_button.icon = ft.icons.PLAY_ARROW_ROUNDED
    
    def _build_step_config(self):



        return ft.Column([

            create_section_header(
                "Konfiguracja Eksperymentu",
                "Ustaw parametry globalne dla zapewnienia reprodukowalności"
            ),
            ft.Container(height=30),


            ft.Container(
                content=ft.Column([

                    ft.Row([
                        ft.Icon(ft.icons.SCIENCE_ROUNDED, color=AppColors.PRIMARY, size=24),
                        ft.Text("Scientific Control", size=16,
                               color=AppColors.TEXT_PRIMARY,
                               weight=ft.FontWeight.W_600),
                    ], spacing=12),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Text("Random Seed", size=14,
                               color=AppColors.TEXT_SECONDARY,
                               weight=ft.FontWeight.W_500),
                        ft.Container(height=8),
                        ft.TextField(
                            value=self.random_seed,
                            hint_text="np. 42",
                            keyboard_type=ft.KeyboardType.NUMBER,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            text_size=14,
                            height=50,
                            on_change=lambda e: self._on_seed_change(e),
                            error_text=self.seed_validation_error,
                        ),
                        ft.Container(height=8),

                        ft.Row([
                            ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED,
                                   color=AppColors.TEXT_MUTED, size=16),
                            ft.Text(
                                "Wymagana liczba całkowita nieujemna. Ustawienie stałego ziarna gwarantuje powtarzalność wyników.",
                                size=12,
                                color=AppColors.TEXT_MUTED,
                                italic=True,
                            ),
                        ], spacing=8),
                    ]),

                    ft.Container(height=20),


                    ft.Column([
                        ft.Text("Walidacja datasetu", size=14,
                               color=AppColors.TEXT_SECONDARY,
                               weight=ft.FontWeight.W_500),
                        ft.Container(height=8),
                        ft.Row([
                            ft.Switch(
                                value=not self.skip_validation,
                                active_color=AppColors.PRIMARY,
                                on_change=lambda e: setattr(self, 'skip_validation', not e.control.value),
                            ),
                            ft.Column([
                                ft.Text("Włącz walidację przed eksperymentem",
                                       size=13, color=AppColors.TEXT_PRIMARY),
                            ], spacing=2, expand=True),
                        ], spacing=12),
                        ft.Container(height=8),
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Co sprawdza walidacja:", size=12, color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                                ft.Text("• Dominacja kolumn kategorycznych (>50% = błąd krytyczny)", size=11, color=AppColors.TEXT_MUTED),
                                ft.Text("• Liczby zapisane jako tekst", size=11, color=AppColors.TEXT_MUTED),
                                ft.Text("• Rozmiar datasetu (<100 wierszy = ostrzeżenie)", size=11, color=AppColors.TEXT_MUTED),
                                ft.Text("• Brakujące wartości (info)", size=11, color=AppColors.TEXT_MUTED),
                                ft.Text("• Niezbalansowane klasy (>10x różnica = ostrzeżenie)", size=11, color=AppColors.TEXT_MUTED),
                                ft.Text("• Kolumny ze stałą wartością", size=11, color=AppColors.TEXT_MUTED),
                                ft.Container(height=4),
                                ft.Text("WYŁĄCZONA: Eksperyment startuje bez sprawdzania danych (szybciej).",
                                       size=11, color=AppColors.TEXT_MUTED, italic=True),
                                ft.Text("WŁĄCZONA: Błędy krytyczne zatrzymają eksperyment przed startem.",
                                       size=11, color=AppColors.TEXT_MUTED, italic=True),
                            ], spacing=2),
                            padding=ft.padding.only(left=48),
                        ),
                    ]),
                ], spacing=0),
                padding=24,
                border_radius=12,
                bgcolor=AppColors.BG_ELEVATED,
                border=ft.border.all(1, AppColors.BORDER),
            ),

            ft.Container(height=24),


            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.TIPS_AND_UPDATES_OUTLINED,
                           color=AppColors.SECONDARY, size=20),
                    ft.Column([
                        ft.Text("Dlaczego Random Seed jest ważny?",
                               size=13, color=AppColors.TEXT_PRIMARY,
                               weight=ft.FontWeight.W_600),
                        ft.Text(
                            "Random seed kontroluje wszystkie operacje losowe (dyskretyzacja, "
                            "generowanie reguł, wybór strategii). Ten sam seed = identyczne wyniki.",
                            size=12, color=AppColors.TEXT_SECONDARY),
                    ], spacing=4, expand=True),
                ], spacing=12),
                padding=16,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.05, AppColors.SECONDARY),
                border=ft.border.all(1, ft.colors.with_opacity(0.2, AppColors.SECONDARY)),
            ),
        ])

    def _on_seed_change(self, e):

        self.random_seed = e.control.value
        is_valid, error_msg = self._validate_seed(e.control.value)

        if is_valid:
            self.seed_validation_error = None
            e.control.border_color = AppColors.BORDER
            e.control.error_text = None
        else:
            self.seed_validation_error = error_msg
            e.control.border_color = ft.colors.RED
            e.control.error_text = error_msg


        e.control.update()

    def _validate_seed(self, value: str) -> tuple[bool, str]:



        try:
            num = int(value.strip())
            if num < 0:
                return False, "Wymagana liczba całkowita nieujemna"
            return True, ""
        except ValueError:
            return False, "Wymagana liczba całkowita nieujemna"

    def _build_step_data(self):

        self.user_files = self.firebase.list_user_files()


        self.file_status_container = ft.Container(
            visible=False,
            padding=12,
            border_radius=8,
        )


        elements = []


        if self.loaded_file_path and self.loaded_metadata:
            elements.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text("Wczytany plik:", size=14,
                               color=AppColors.TEXT_SECONDARY,
                               weight=ft.FontWeight.W_600),
                        ft.Container(height=8),
                        ft.Row([
                            ft.Icon(ft.icons.DESCRIPTION_ROUNDED,
                                   color=AppColors.SECONDARY, size=20),
                            ft.Column([
                                ft.Text(self.loaded_metadata['filename'],
                                       size=14,
                                       color=AppColors.TEXT_PRIMARY,
                                       weight=ft.FontWeight.W_500),
                                ft.Text(self.loaded_file_path,
                                       size=12,
                                       color=AppColors.TEXT_MUTED,
                                       italic=True),
                            ], spacing=2, expand=True),
                        ], spacing=12),
                    ], spacing=0),
                    padding=16,
                    border_radius=12,
                    bgcolor=AppColors.BG_ELEVATED,
                    border=ft.border.all(1, AppColors.SECONDARY),
                )
            )
            elements.append(ft.Container(height=20))


        if self.loaded_file_path:
            header_title = "Wybierz inny plik jako dane wejściowe"
            header_subtitle = "Lub wybierz z przykładowych zestawów danych które znajdziesz poniżej:"
        else:
            header_title = "Wybierz dane wejściowe"
            header_subtitle = "Załaduj plik CSV z danymi lub wybierz z przykładowych"

        elements.append(
            create_section_header(header_title, header_subtitle)
        )
        elements.append(ft.Container(height=20))


        elements.append(
            ft.GestureDetector(
                content=ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.CLOUD_UPLOAD_ROUNDED, size=48, color=AppColors.PRIMARY),
                        ft.Text("Przeciągnij plik CSV tutaj", size=16,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ft.Text("lub kliknij aby wybrać", size=13, color=AppColors.TEXT_MUTED),
                        ft.Container(height=10),
                        ft.OutlinedButton(
                            "Wybierz plik",
                            style=ft.ButtonStyle(color=AppColors.PRIMARY),
                            on_click=lambda _: self.file_picker.pick_files(
                                allowed_extensions=["csv", "txt"],
                                dialog_title="Wybierz plik CSV",
                            )
                        ),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
                    padding=40,
                    border_radius=12,
                    border=ft.border.all(2, AppColors.BORDER),
                    bgcolor=AppColors.BG_ELEVATED,
                    alignment=ft.alignment.center,
                ),
                on_tap=lambda _: self.file_picker.pick_files(
                    allowed_extensions=["csv", "txt"],
                    dialog_title="Wybierz plik CSV",
                ),
            )
        )

        elements.append(ft.Container(height=10))

        elements.append(self.file_status_container)

        elements.append(ft.Container(height=20))
        elements.append(ft.Divider(color=AppColors.BORDER))
        elements.append(ft.Container(height=20))


        elements.append(
            ft.Text("Pliki w pamięci programu", size=14,
                   color=AppColors.TEXT_SECONDARY)
        )
        elements.append(ft.Container(height=10))
        elements.append(self._build_recent_files_section())

        elements.append(ft.Container(height=20))
        elements.append(ft.Divider(color=AppColors.BORDER))
        elements.append(ft.Container(height=20))


        elements.append(
            ft.Text(lang.t('data_network_locations'), size=14,
                   color=AppColors.TEXT_SECONDARY)
        )
        elements.append(ft.Container(height=10))
        elements.append(self._build_firebase_files_section())

        elements.append(ft.Container(height=20))
        elements.append(ft.Divider(color=AppColors.BORDER))
        elements.append(ft.Container(height=20))

        elements.append(
            ft.Text("Lub wybierz przykładowy dataset:", size=14,
                   color=AppColors.TEXT_SECONDARY)
        )
        elements.append(ft.Container(height=10))

        elements.append(
            ft.Row([
                self._create_dataset_option("Wine", "178 rekordów, 13 atrybutów"),
                self._create_dataset_option("Mushroom", "8124 rekordów, 22 atrybuty"),
                self._create_dataset_option("Iris", "150 rekordów, 4 atrybuty"),
            ], spacing=12)
        )

        elements.append(ft.Container(height=10))


        elements.append(
            ft.Row([
                self._create_dataset_option("Breast Cancer", "699 rekordów, 11 atrybutów"),
                self._create_dataset_option("Zoo", "101 rekordów, 18 atrybutów"),
                self._create_dataset_option("Income", "32561 rekordów, 15 atrybutów"),
            ], spacing=12)
        )

        elements.append(ft.Container(height=10))


        elements.append(
            ft.Row([
                self._create_dataset_option("Car", "1728 rekordów, 7 atrybutów"),
                self._create_dataset_option("Indians Diabetes", "768 rekordów, 9 atrybutów"),
                self._create_dataset_option("Ecoli", "336 rekordów, 9 atrybutów"),
            ], spacing=12)
        )

        return ft.Column(elements)

    def _build_step_csv_config(self):



        if self.loaded_df is not None and self.loaded_metadata is not None:
            filename = self.loaded_metadata['filename']
            rows = self.loaded_metadata['rows_final']
            cols = self.loaded_metadata['columns_total']

            return ft.Column([
                create_section_header(lang.t('csv_config_title')),
                ft.Container(height=20),


                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.CHECK_CIRCLE_ROUNDED, color=AppColors.SECONDARY, size=48),
                        ft.Container(height=12),
                        ft.Text(
                            "Plik został poprawnie wczytany!",
                            size=16,
                            color=AppColors.TEXT_PRIMARY,
                            weight=ft.FontWeight.W_600
                        ),
                        ft.Container(height=8),
                        ft.Text(
                            f"'{filename}' - {rows} wierszy, {cols} kolumn",
                            size=13,
                            color=AppColors.TEXT_SECONDARY
                        ),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=40,
                    border_radius=12,
                    bgcolor=AppColors.SECONDARY_BG,
                    border=ft.border.all(2, AppColors.SECONDARY),
                ),

                ft.Container(height=30),


                ft.Row([
                    ft.ElevatedButton(
                        "Przejdź do Imputacji (Dalej)",
                        icon=ft.icons.ARROW_FORWARD_ROUNDED,
                        on_click=lambda _: self._go_next(),
                        style=ft.ButtonStyle(
                            bgcolor=AppColors.PRIMARY,
                            color="white",
                            padding=ft.padding.symmetric(horizontal=24, vertical=16),
                        ),
                    )
                ], alignment=ft.MainAxisAlignment.END),
            ])


        if self.loaded_file_path and len(self.csv_preview_lines) > 0:
            first_line = self.csv_preview_lines[0]

            detected_cols = first_line.split(self.csv_column_separator)
            self.csv_detected_columns = [col.strip() for col in detected_cols]


            if self.csv_decision_column is None and len(self.csv_detected_columns) > 0:

                decision_column_patterns = [
                    'class', 'klasa', 'klasa_decyzyjna', 'klasa decyzyjna',
                    'decision class', 'decision_class', 'decyzja', 'decision',
                    'target', 'label', 'result', 'wynik'
                ]


                found_decision_column = None
                for col in self.csv_detected_columns:
                    col_lower = col.lower().strip()
                    if col_lower in decision_column_patterns:
                        found_decision_column = col
                        break


                if found_decision_column:
                    self.csv_decision_column = found_decision_column
                else:
                    self.csv_decision_column = self.csv_detected_columns[-1]


        if not self.loaded_file_path:
            return ft.Column([
                create_section_header(lang.t('csv_config_title')),
                ft.Container(height=20),
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, size=48, color=AppColors.WARNING),
                        ft.Text(lang.t('csv_no_file_selected'), size=16, color=AppColors.TEXT_SECONDARY),
                        ft.Text(lang.t('csv_go_back'),
                               size=13, color=AppColors.TEXT_MUTED),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
                    padding=60,
                    alignment=ft.alignment.center,
                ),
            ])


        preview_text = "\n".join(self.csv_preview_lines) if self.csv_preview_lines else "Brak podglądu"


        column_sep_dropdown = ft.Dropdown(
            label="Separator kolumn",
            value=self.csv_column_separator,
            options=[
                ft.dropdown.Option(",", "Przecinek (,)"),
                ft.dropdown.Option(";", "Średnik (;)"),
                ft.dropdown.Option("\t", "Tabulator (Tab)"),
                ft.dropdown.Option(" ", "Spacja ( )"),
            ],
            on_change=lambda e: setattr(self, 'csv_column_separator', e.control.value),
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )

        decimal_sep_dropdown = ft.Dropdown(
            label="Separator dziesiętny",
            value=self.csv_decimal_separator,
            options=[
                ft.dropdown.Option(".", "Kropka (.)"),
                ft.dropdown.Option(",", "Przecinek (,)"),
            ],
            on_change=lambda e: setattr(self, 'csv_decimal_separator', e.control.value),
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )

        has_header_switch = ft.Switch(
            label="Pierwsza linia to nagłówki",
            value=self.csv_has_header,
            active_color=AppColors.PRIMARY,
            on_change=lambda e: setattr(self, 'csv_has_header', e.control.value),
        )


        decision_column_dropdown = ft.Dropdown(
            label=lang.t('csv_config_decision_column_label'),
            value=self.csv_decision_column,
            options=[
                ft.dropdown.Option(col, col)
                for col in self.csv_detected_columns
            ] if self.csv_detected_columns else [],
            on_change=lambda e: setattr(self, 'csv_decision_column', e.control.value),
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )


        accept_auto_button = ft.ElevatedButton(
            lang.t('csv_config_accept_auto'),
            icon=ft.icons.ARROW_FORWARD_ROUNDED,
            style=ft.ButtonStyle(
                bgcolor=AppColors.SECONDARY,
                color=AppColors.TEXT_PRIMARY,
                padding=ft.padding.symmetric(horizontal=24, vertical=12),
            ),
            on_click=lambda _: self._validate_csv_with_config(),
        )


        validate_manual_button = ft.ElevatedButton(
            lang.t('csv_config_validate_manual'),
            icon=ft.icons.CHECK_CIRCLE_ROUNDED,
            style=ft.ButtonStyle(
                bgcolor=AppColors.PRIMARY,
                color=AppColors.TEXT_PRIMARY,
                padding=ft.padding.symmetric(horizontal=24, vertical=12),
            ),
            on_click=lambda _: self._validate_csv_with_config(),
        )

        return ft.Column([
            create_section_header(lang.t('csv_config_title')),
            ft.Container(height=20),


            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.DESCRIPTION_ROUNDED, color=AppColors.PRIMARY, size=20),
                    ft.Text(f"{lang.t('csv_file_prefix')} {self.loaded_file_path}", size=13,
                           color=AppColors.TEXT_SECONDARY),
                ], spacing=8),
                padding=12,
                border_radius=8,
                bgcolor=AppColors.BG_ELEVATED,
            ),

            ft.Container(height=20),


            ft.Text(lang.t('csv_preview_raw'), size=14,
                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_600),
            ft.Container(height=8),
            ft.Container(
                content=ft.Text(
                    preview_text,
                    size=12,
                    color=AppColors.TEXT_PRIMARY,
                    font_family=AppFonts.MONO,
                    selectable=True,
                ),
                padding=12,
                border_radius=8,
                bgcolor=AppColors.BG_DARK,
                border=ft.border.all(1, AppColors.BORDER),
            ),

            ft.Container(height=30),


            ft.Text(lang.t('csv_config_params'), size=14,
                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_600),
            ft.Container(height=12),


            ft.Text(lang.t('csv_config_auto_detected'), size=13,
                   color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
            ft.Container(height=8),
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text(lang.t('csv_config_col_separator'), size=13,
                               color=AppColors.TEXT_SECONDARY),
                        ft.Text(f'"{self.csv_column_separator}"' if self.csv_column_separator != "\t" else '"\\t" (Tab)',
                               size=13, color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=8),
                    ft.Row([
                        ft.Text(lang.t('csv_config_decimal_separator'), size=13,
                               color=AppColors.TEXT_SECONDARY),
                        ft.Text(f'"{self.csv_decimal_separator}"', size=13,
                               color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=8),
                    ft.Row([
                        ft.Text(lang.t('csv_config_first_row_headers'), size=13,
                               color=AppColors.TEXT_SECONDARY),
                        ft.Text(lang.t('csv_config_yes') if self.csv_has_header else lang.t('csv_config_no'),
                               size=13, color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=8),
                    ft.Row([
                        ft.Text(lang.t('csv_config_decision_column'), size=13,
                               color=AppColors.TEXT_SECONDARY),
                        ft.Text(f'"{self.csv_decision_column}"', size=13,
                               color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=8),
                ], spacing=6),
                padding=12,
                border_radius=8,
                bgcolor=AppColors.BG_ELEVATED,
            ),

            ft.Container(height=20),


            accept_auto_button,

            ft.Container(height=20),


            ft.Text(lang.t('csv_config_manual'), size=13,
                   color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
            ft.Container(height=12),

            ft.Row([
                column_sep_dropdown,
                decimal_sep_dropdown,
                decision_column_dropdown,
            ], spacing=20, wrap=True),

            ft.Container(height=12),
            has_header_switch,

            ft.Container(height=30),


            validate_manual_button,

            ft.Container(height=10),

            self.file_status_container if hasattr(self, 'file_status_container') else ft.Container(),
        ])

    def _build_firebase_files_section(self):


        if not self.firebase.is_logged_in():
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.WARNING, size=20),
                    ft.Text("Zaloguj się w zakładce Ustawienia aby zobaczyć swoje pliki",
                           size=13, color=AppColors.TEXT_SECONDARY),
                ], spacing=8),
                padding=16,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.1, AppColors.WARNING),
            )


        if not self.user_files:
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.FOLDER_OPEN_ROUNDED, color=AppColors.TEXT_MUTED, size=20),
                    ft.Text("Brak plików w chmurze",
                           size=13, color=AppColors.TEXT_MUTED),
                ], spacing=8),
                padding=16,
                border_radius=8,
                bgcolor=AppColors.BG_ELEVATED,
            )


        file_widgets = []
        for file_data in self.user_files:
            size_kb = file_data['size'] / 1024


            is_selected = (self.loaded_file_path and file_data['filename'] in self.loaded_file_path)

            file_widgets.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.DESCRIPTION_ROUNDED,
                               color=AppColors.PRIMARY if is_selected else AppColors.SECONDARY,
                               size=20),
                        ft.Column([
                            ft.Text(file_data['filename'], size=13,
                                   color=AppColors.TEXT_PRIMARY,
                                   weight=ft.FontWeight.BOLD if is_selected else ft.FontWeight.W_600),
                            ft.Text(f"{size_kb:.2f} KB", size=11,
                                   color=AppColors.TEXT_MUTED),
                        ], spacing=2, expand=True),
                    ], spacing=12),
                    padding=12,
                    border_radius=8,
                    bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if is_selected else AppColors.BG_ELEVATED,
                    border=ft.border.all(2, AppColors.PRIMARY if is_selected else AppColors.BORDER),
                    on_click=lambda e, fd=file_data: self._load_firebase_file(fd),
                    ink=True,
                )
            )

        return ft.Column(file_widgets, spacing=8)

    def _build_recent_files_section(self):





        files_with_last_used = [
            f for f in self.local_files
            if 'last_used' in f and f['last_used']
        ]


        if not files_with_last_used:
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.HISTORY_ROUNDED, color=AppColors.TEXT_MUTED, size=20),
                    ft.Text("Brak ostatnio używanych plików",
                           size=13, color=AppColors.TEXT_MUTED),
                ], spacing=8),
                padding=16,
                border_radius=8,
                bgcolor=AppColors.BG_ELEVATED,
            )


        sorted_files = sorted(
            files_with_last_used,
            key=lambda f: f['last_used'],
            reverse=True
        )


        recent_files = sorted_files[:5]


        valid_files = []
        files_to_remove = []
        for file_data in recent_files:
            file_path = file_data['path']

            if os.path.exists(file_path):
                valid_files.append(file_data)
            else:

                files_to_remove.append(file_data)
                print(f"[RECENT FILES] Plik nie istnieje (usunę): {file_path}")


        if files_to_remove:
            for file_to_remove in files_to_remove:
                self.local_files = [f for f in self.local_files if f['path'] != file_to_remove['path']]
            self._save_local_files()


        if not valid_files:
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.HISTORY_ROUNDED, color=AppColors.TEXT_MUTED, size=20),
                    ft.Text("Brak ostatnio używanych plików",
                           size=13, color=AppColors.TEXT_MUTED),
                ], spacing=8),
                padding=16,
                border_radius=8,
                bgcolor=AppColors.BG_ELEVATED,
            )


        file_widgets = []
        for file_data in valid_files:
            file_name = file_data['name']
            file_path = file_data['path']


            is_selected = (file_path == self.loaded_file_path)

            file_widgets.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.DESCRIPTION_ROUNDED,
                               color=AppColors.PRIMARY if is_selected else AppColors.SECONDARY,
                               size=20),
                        ft.Column([
                            ft.Text(file_name, size=13,
                                   color=AppColors.TEXT_PRIMARY,
                                   weight=ft.FontWeight.BOLD if is_selected else ft.FontWeight.W_600),
                            ft.Text(file_path, size=11,
                                   color=AppColors.TEXT_MUTED,
                                   italic=True),
                        ], spacing=2, expand=True),
                    ], spacing=12),
                    padding=12,
                    border_radius=8,
                    bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if is_selected else AppColors.BG_ELEVATED,
                    border=ft.border.all(2, AppColors.PRIMARY if is_selected else AppColors.BORDER),
                    on_click=lambda e, path=file_path: self._load_recent_file(path),
                    ink=True,
                )
            )

        return ft.Column(file_widgets, spacing=8)

    def _load_recent_file(self, file_path: str):


        if not os.path.exists(file_path):
            print(f"[ERROR] Plik nie istnieje: {file_path}")

            self.local_files = [f for f in self.local_files if f['path'] != file_path]
            self._save_local_files()

            self._update_content()
            self.update()
            return


        self.selected_dataset = None


        self.loaded_file_path = file_path


        self._add_or_update_local_file(file_path)


        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='windows-1250') as f:
                    self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
            except Exception:
                self.csv_preview_lines = ["Nie można odczytać podglądu pliku"]
        except Exception as e:
            self.csv_preview_lines = [f"Błąd odczytu: {str(e)}"]


        self.current_step = 1
        self._update_stepper()
        self._update_content()
        self._update_navigation_buttons()
        self.update()

    def _load_firebase_file(self, file_data: dict):






        print(f"[FIREBASE] Pobieranie pliku: {file_data['filename']}")


        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_firebase')
        os.makedirs(temp_dir, exist_ok=True)


        temp_file_path = os.path.join(temp_dir, file_data['filename'])


        success = self.firebase.download_file(file_data['id'], temp_file_path)

        if not success:
            print(f"[ERROR] Nie udało się pobrać pliku z Firebase")

            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text("Błąd pobierania pliku z chmury", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return


        self.selected_dataset = None


        self.loaded_file_path = temp_file_path


        self._add_or_update_local_file(temp_file_path)


        try:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
        except UnicodeDecodeError:
            try:
                with open(temp_file_path, 'r', encoding='windows-1250') as f:
                    self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
            except Exception:
                self.csv_preview_lines = ["Nie można odczytać podglądu pliku"]
        except Exception as e:
            self.csv_preview_lines = [f"Błąd odczytu: {str(e)}"]


        self.file_status_container.content = ft.Row([
            ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
            ft.Text(f"Pobrano plik: {file_data['filename']}", size=13, color=AppColors.SECONDARY),
        ], spacing=8)
        self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
        self.file_status_container.visible = True


        self.current_step = 1
        self._update_stepper()
        self._update_content()
        self._update_navigation_buttons()
        self.update()

        print(f"[FIREBASE] Plik wczytany: {temp_file_path}")

    def _auto_load_file(self, file_path: str):





        if not os.path.exists(file_path):
            print(f"[ERROR] Plik nie istnieje: {file_path}")
            return

        print(f"[AUTO-LOAD] Ładuję plik: {file_path}")


        self.loaded_file_path = file_path


        self._add_or_update_local_file(file_path)


        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='windows-1250') as f:
                    self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
            except Exception:
                self.csv_preview_lines = ["Nie można odczytać podglądu pliku"]
        except Exception as e:
            self.csv_preview_lines = [f"Błąd odczytu: {str(e)}"]


        self.current_step = 1
        self._update_stepper()
        self._update_content()
        self._update_navigation_buttons()

    def _load_local_files(self) -> list:

        if os.path.exists(self.local_files_path):
            try:
                with open(self.local_files_path, 'r', encoding='utf-8') as f:
                    files = json.load(f)


                migrated = False
                for file in files:
                    if 'last_used' not in file:

                        file['last_used'] = file.get('date_added', '')
                        migrated = True


                if migrated:
                    with open(self.local_files_path, 'w', encoding='utf-8') as f:
                        json.dump(files, f, indent=2, ensure_ascii=False)
                    print(f"[NEW_EXP] Migracja: Dodano pole last_used do {len(files)} plików")

                print(f"[NEW_EXP] Wczytano {len(files)} lokalnych plików")
                return files
            except Exception as e:
                print(f"[ERROR] Błąd wczytywania lokalnych plików: {e}")
                return []
        else:
            print("[NEW_EXP] Brak local_files.json - pusta lista")
            return []

    def _save_local_files(self):

        try:
            with open(self.local_files_path, 'w', encoding='utf-8') as f:
                json.dump(self.local_files, f, indent=2, ensure_ascii=False)
            print(f"[NEW_EXP] Zapisano {len(self.local_files)} plików do local_files.json")
        except Exception as e:
            print(f"[ERROR] Błąd zapisywania lokalnych plików: {e}")

    def _add_or_update_local_file(self, file_path: str):




        import datetime

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        size_mb = f"{file_size / (1024 * 1024):.1f} MB"
        ext = os.path.splitext(file_name)[1]
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        existing_file = None
        for file in self.local_files:
            if file['path'] == file_path:
                existing_file = file
                break

        if existing_file:

            existing_file['last_used'] = now
            print(f"[NEW_EXP] Zaktualizowano last_used: {file_name}")
        else:

            new_file = {
                "name": file_name,
                "path": file_path,
                "size": size_mb,
                "ext": ext,
                "attrs": "?",
                "rows": "?",
                "date_added": now,
                "last_used": now,
            }
            self.local_files.append(new_file)
            print(f"[NEW_EXP] Dodano nowy plik: {file_name}")


        self._save_local_files()

    def _create_dataset_option(self, name: str, desc: str):
        selected = name == self.selected_dataset
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.TABLE_CHART_ROUNDED,
                           color=AppColors.PRIMARY if selected else AppColors.TEXT_MUTED, size=20),
                    ft.Text(name, size=14, color=AppColors.TEXT_PRIMARY,
                           weight=ft.FontWeight.W_600),
                ], spacing=8),
                ft.Text(desc, size=12, color=AppColors.TEXT_MUTED),
            ], spacing=4),
            padding=16,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if selected else AppColors.BG_ELEVATED,
            border=ft.border.all(2, AppColors.PRIMARY if selected else AppColors.BORDER),
            expand=True,
            on_click=lambda e, dataset=name: self._select_dataset(dataset),
            ink=True,
        )

    def _build_step_imputation(self):




        return ft.Column([
            create_section_header(lang.t('imputation_title'),
                                lang.t('imputation_subtitle')),
            ft.Container(height=20),

            ft.Row([
                self._create_imputation_button(
                    lang.t('imputation_remove_rows'),
                    lang.t('imputation_remove_rows_sub'),
                    ft.icons.DELETE_OUTLINE_ROUNDED,
                    "remove_rows"
                ),
                self._create_imputation_button(
                    lang.t('imputation_smart_fill'),
                    lang.t('imputation_smart_fill_hint'),
                    ft.icons.AUTO_FIX_HIGH_ROUNDED,
                    "smart_fill",
                    tooltip=lang.t('imputation_smart_fill_tooltip')
                ),
            ], spacing=16),

            ft.Container(height=24),


            ft.Container(
                content=ft.Column([
                    ft.Text("Konfiguracja Smart Fill", size=14,
                           color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                    ft.Container(height=12),


                    ft.Column([
                        ft.Text("Wartości numeryczne:", size=13,
                               color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                        ft.Container(height=8),
                        ft.RadioGroup(
                            content=ft.Column([
                                ft.Radio(
                                    value="mean",
                                    label="Mean (średnia per klasa)",
                                    fill_color=AppColors.PRIMARY,
                                ),
                                ft.Radio(
                                    value="median",
                                    label="Median (mediana per klasa)",
                                    fill_color=AppColors.PRIMARY,
                                ),
                            ], spacing=8),
                            value=self.imputation_numeric_method,
                            on_change=lambda e: setattr(self, 'imputation_numeric_method', e.control.value),
                        ),
                    ]),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Text("Wartości kategoryczne:", size=13,
                               color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                        ft.Container(height=8),
                        ft.RadioGroup(
                            content=ft.Column([
                                ft.Radio(
                                    value="mode",
                                    label="Mode (najczęstsza wartość per klasa)",
                                    fill_color=AppColors.PRIMARY,
                                ),
                            ], spacing=8),
                            value=self.imputation_categorical_method,
                            on_change=lambda e: setattr(self, 'imputation_categorical_method', e.control.value),
                        ),
                    ]),
                ], spacing=0),
                padding=20,
                border_radius=12,
                bgcolor=AppColors.BG_ELEVATED,
                border=ft.border.all(1, AppColors.BORDER),
                visible=(self.selected_imputation == "smart_fill"),
            ),

            ft.Container(height=24),


            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.PRIMARY, size=20),
                        ft.Text(
                            "Imputacja brakujących wartości",
                            size=14,
                            color=AppColors.TEXT_PRIMARY,
                            weight=ft.FontWeight.W_600
                        ),
                    ], spacing=8),
                    ft.Container(height=8),
                    ft.Text(
                        "Wybierz metodę obsługi brakujących danych (NaN). To jest wymagany krok przed dyskretyzacją.",
                        size=12,
                        color=AppColors.TEXT_SECONDARY
                    ),
                ]),
                padding=16,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.05, AppColors.PRIMARY),
                border=ft.border.all(1, ft.colors.with_opacity(0.2, AppColors.PRIMARY)),
            ),
        ])

    def _build_step_discretization(self):

        has_numeric_columns = False
        if self.loaded_df is not None:
            for col in self.loaded_df.columns:
                if col != self.csv_decision_column and pd.api.types.is_numeric_dtype(self.loaded_df[col]):
                    has_numeric_columns = True
                    break

        return ft.Column([
            create_section_header("Metoda dyskretyzacji",
                                "Wybierz sposób podziału wartości ciągłych na kategorie"),
            ft.Container(height=20),

            ft.Row([
                self._create_method_card(
                    "Equal Width",
                    "Równe szerokości przedziałów",
                    "szerokość = (max - min) / n_bins",
                    ft.icons.STRAIGHTEN_ROUNDED
                ),
                self._create_method_card(
                    "Equal Frequency",
                    "Równa liczba obserwacji w binach",
                    "Każdy bin ma tyle samo danych, dane dzielimy tak aby każdy przedział zawierał taką samą liczbę obserwacji.",
                    ft.icons.EQUALIZER_ROUNDED
                ),
                self._create_method_card(
                    "K-Means",
                    "Clustering jako dyskretyzacja",
                    lang.t('disc_kmeans_detail'),
                    ft.icons.HUB_ROUNDED
                ),
                self._create_method_card(
                    "Brak",
                    "Pomiń dyskretyzację",
                    "Dane już są kategoryczne",
                    ft.icons.BLOCK_ROUNDED
                ),
            ], spacing=12),

            ft.Container(height=20),


            self._build_smart_binning_advisor(),

            ft.Container(height=30),


            ft.Row([
                self._create_bins_choice_card(
                    lang.t('disc_auto_title'),
                    lang.t('disc_auto_subtitle'),
                    ft.icons.AUTO_MODE_ROUNDED,
                    "auto",
                    disabled_no_numeric=not has_numeric_columns
                ),
                self._create_bins_choice_card(
                    lang.t('disc_manual_title'),
                    lang.t('disc_manual_subtitle'),
                    ft.icons.TUNE_ROUNDED,
                    "manual",
                    disabled_no_numeric=False
                ),
            ], spacing=16),
        ])
    
    def _build_smart_binning_advisor(self):





        recommended_bins = self._calculate_recommended_bins()


        if self.loaded_df is not None and self.bin_suggestion is not None:
            N = len(self.loaded_df)
            has_data = True
            suggestion = self.bin_suggestion
        else:
            N = 0
            has_data = False
            suggestion = None


        if has_data and suggestion:

            method_names = {
                "sturges": "Sturges",
                "scott": "Scott",
                "freedman_diaconis": "Freedman-Diaconis"
            }
            method_name = method_names.get(suggestion.recommended, suggestion.recommended)

            recommendation_text = f"Rekomendujemy {recommended_bins} binów (Metoda: {method_name})"


            methods_text = f"Sturges: {suggestion.sturges} | Scott: {suggestion.scott} | FD: {suggestion.freedman_diaconis}"


            reasons_text = "Powody: " + ", ".join(suggestion.reasons)

            return ft.Container(
                content=ft.Column([

                    ft.Row([
                        ft.Icon(ft.icons.LIGHTBULB_OUTLINE_ROUNDED,
                               color=AppColors.SECONDARY,
                               size=24),
                        ft.Text("Smart Binning Advisor (System Ekspertowy)", size=14,
                               color=AppColors.TEXT_PRIMARY,
                               weight=ft.FontWeight.W_600),
                    ], spacing=8),

                    ft.Container(height=12),


                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.icons.STARS_ROUNDED, color=AppColors.SECONDARY, size=20),
                            ft.Text(recommendation_text, size=15,
                                   color=AppColors.SECONDARY,
                                   weight=ft.FontWeight.W_600),
                        ], spacing=8),
                        padding=12,
                        border_radius=8,
                        bgcolor=AppColors.SECONDARY_BG,
                    ),

                    ft.Container(height=12),


                    ft.Column([
                        ft.Text("Porównanie metod:", size=12,
                               color=AppColors.TEXT_SECONDARY,
                               weight=ft.FontWeight.W_500),
                        ft.Container(height=4),
                        ft.Text(methods_text, size=11,
                               color=AppColors.TEXT_MUTED,
                               italic=True),
                    ], spacing=0),

                    ft.Container(height=8),


                    ft.Column([
                        ft.Row([
                            ft.Icon(ft.icons.STORAGE_ROUNDED, color=AppColors.PRIMARY, size=16),
                            ft.Text("Uzasadnienie (XAI):", size=12,
                                   color=AppColors.TEXT_SECONDARY,
                                   weight=ft.FontWeight.W_500),
                        ], spacing=4),
                        ft.Container(height=4),
                        ft.Text(reasons_text, size=11,
                               color=AppColors.TEXT_MUTED),
                    ], spacing=0),

                    ft.Container(height=8),


                    ft.Text(f"Analizowany dataset: {N} wierszy", size=10,
                           color=AppColors.TEXT_MUTED,
                           italic=True),
                ], spacing=0),
                padding=20,
                border_radius=12,
                bgcolor=ft.colors.with_opacity(0.05, AppColors.SECONDARY),
                border=ft.border.all(2, AppColors.SECONDARY),
            )
        elif self.loaded_df is not None and self.bin_suggestion is None:

            N = len(self.loaded_df)

            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED,
                               color=ft.colors.ORANGE,
                               size=20),
                        ft.Text("Brak rekomendacji", size=14,
                               color=AppColors.TEXT_PRIMARY,
                               weight=ft.FontWeight.W_600),
                    ], spacing=8),
                    ft.Container(height=8),
                    ft.Text(
                        "W wykrytym zbiorze danych nie znaleziono kolumn numerycznych wymagających dyskretyzacji.",
                        size=13,
                        color=AppColors.TEXT_SECONDARY
                    ),
                    ft.Container(height=4),
                    ft.Text(
                        "Możesz pominąć ten krok lub wybrać metodę ręcznie.",
                        size=11,
                        color=AppColors.TEXT_MUTED,
                        italic=True
                    ),
                    ft.Container(height=8),
                    ft.Text(
                        f"Dataset: {N} wierszy (tylko kolumny kategoryczne)",
                        size=10,
                        color=AppColors.TEXT_MUTED,
                        italic=True
                    ),
                ], spacing=0),
                padding=16,
                border_radius=12,
                bgcolor=ft.colors.with_opacity(0.05, ft.colors.ORANGE),
                border=ft.border.all(1, ft.colors.ORANGE),
            )
        else:

            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.icons.LIGHTBULB_OUTLINE_ROUNDED,
                               color=AppColors.TEXT_MUTED,
                               size=20),
                        ft.Text("Smart Binning Advisor", size=14,
                               color=AppColors.TEXT_SECONDARY,
                               weight=ft.FontWeight.W_600),
                    ], spacing=8),
                    ft.Container(height=8),
                    ft.Text(
                        "Wczytaj plik CSV aby zobaczyć rekomendację",
                        size=13,
                        color=AppColors.TEXT_MUTED
                    ),
                    ft.Container(height=4),
                    ft.Text(
                        "System ekspertowy przeanalizuje Twoje dane i zaproponuje optymalną liczbę binów",
                        size=11,
                        color=AppColors.TEXT_MUTED,
                        italic=True
                    ),
                ], spacing=0),
                padding=16,
                border_radius=12,
                bgcolor=AppColors.BG_ELEVATED,
                border=ft.border.all(1, AppColors.BORDER),
            )

    def _create_method_card(self, title: str, subtitle: str, detail: str, icon):
        selected = title == self.selected_discretization
        return ft.Container(
            content=ft.Column([
                ft.Icon(icon, size=32,
                       color=AppColors.PRIMARY if selected else AppColors.TEXT_MUTED),
                ft.Text(title, size=14, color=AppColors.TEXT_PRIMARY,
                       weight=ft.FontWeight.W_600, text_align=ft.TextAlign.CENTER),
                ft.Text(subtitle, size=11, color=AppColors.TEXT_SECONDARY,
                       text_align=ft.TextAlign.CENTER),
                ft.Container(height=4),
                ft.Text(detail, size=10, color=AppColors.TEXT_MUTED, italic=True,
                       text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
            padding=16,
            border_radius=12,
            bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if selected else AppColors.BG_ELEVATED,
            border=ft.border.all(2, AppColors.PRIMARY if selected else AppColors.BORDER),
            expand=True,
            height=160,
            on_click=lambda e, method=title: self._select_discretization(method),
            ink=True,
        )

    def _create_imputation_button(self, title: str, subtitle: str, icon, method_id: str, tooltip: str = None):

        selected = method_id == self.selected_imputation

        button_content = ft.Container(
            content=ft.Column([
                ft.Icon(icon, size=32,
                       color=AppColors.PRIMARY if selected else AppColors.TEXT_MUTED),
                ft.Text(title, size=14, color=AppColors.TEXT_PRIMARY,
                       weight=ft.FontWeight.W_600, text_align=ft.TextAlign.CENTER),
                ft.Text(subtitle, size=11, color=AppColors.TEXT_SECONDARY,
                       text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
            padding=20,
            border_radius=12,
            bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if selected else AppColors.BG_ELEVATED,
            border=ft.border.all(2, AppColors.PRIMARY if selected else AppColors.BORDER),
            expand=True,
            height=120,
            on_click=lambda e, m=method_id: self._select_imputation(m),
            ink=True,
        )


        if tooltip:
            return ft.Tooltip(
                message=tooltip,
                content=button_content,
                text_style=ft.TextStyle(size=12),
                padding=12,
                border_radius=8,
                wait_duration=300,
            )
        else:
            return button_content

    def _create_bins_choice_card(self, title: str, subtitle: str, icon, choice_id: str, disabled_no_numeric: bool = False):

        selected = choice_id == self.bins_choice

        disabled = self.selected_discretization == "Brak" or disabled_no_numeric


        extra_text = None
        if disabled_no_numeric and choice_id == "auto":
            extra_text = "(brak kolumn numerycznych)"

        return ft.Container(
            content=ft.Column([
                ft.Icon(icon, size=40,
                       color=AppColors.PRIMARY if (selected and not disabled) else AppColors.TEXT_MUTED),
                ft.Container(height=8),
                ft.Text(title, size=15, color=AppColors.TEXT_PRIMARY if not disabled else AppColors.TEXT_MUTED,
                       weight=ft.FontWeight.W_600, text_align=ft.TextAlign.CENTER),
                ft.Container(height=4),
                ft.Text(subtitle, size=12, color=AppColors.TEXT_SECONDARY if not disabled else AppColors.TEXT_MUTED,
                       text_align=ft.TextAlign.CENTER),

                ft.Text(extra_text, size=10, color=AppColors.WARNING, italic=True,
                       text_align=ft.TextAlign.CENTER) if extra_text else ft.Container(),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
            padding=24,
            border_radius=12,
            bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if (selected and not disabled) else AppColors.BG_ELEVATED,
            border=ft.border.all(2, AppColors.PRIMARY if (selected and not disabled) else AppColors.BORDER),
            expand=True,
            height=160,
            on_click=lambda e, choice=choice_id: self._select_bins_choice(choice) if not disabled else None,
            ink=not disabled,
            opacity=1.0 if not disabled else 0.5,
        )

    def _build_step_disc_details(self):





        columns_for_discretization = []

        if self.loaded_df is not None:



            for col in self.loaded_df.columns:

                if col == self.csv_decision_column:
                    continue


                if pd.api.types.is_numeric_dtype(self.loaded_df[col]):
                    columns_for_discretization.append(col)

            print(f"[STEP 4] Kolumny numeryczne do dyskretyzacji: {columns_for_discretization}")


            self.available_columns = columns_for_discretization



            if not self.disc_details_initialized and columns_for_discretization:
                self.selected_columns = set(columns_for_discretization)
                self.disc_details_initialized = True
                print(f"[STEP 4] Domyślnie zaznaczono wszystkie kolumny numeryczne (pierwsze wejście)")


        has_numeric_columns = len(columns_for_discretization) > 0


        elements = [
            create_section_header(lang.t('new_exp_step_disc_details'),
                                "Dostosuj parametry dyskretyzacji"),
            ft.Container(height=30),


            ft.Column([
                ft.Text(lang.t('disc_bins_count'), size=14,
                       color=AppColors.TEXT_SECONDARY if has_numeric_columns else AppColors.TEXT_MUTED,
                       weight=ft.FontWeight.W_600),
                ft.Container(height=12),
                ft.Slider(
                    min=2, max=20, divisions=18, value=self.n_bins,
                    active_color=AppColors.PRIMARY if has_numeric_columns else AppColors.TEXT_MUTED,
                    inactive_color=AppColors.BG_ELEVATED,
                    label="{value}",
                    on_change=lambda e: setattr(self, 'n_bins', int(e.control.value)) if has_numeric_columns else None,
                    disabled=not has_numeric_columns,
                ),

                ft.Text(
                    "Slider zablokowany - brak kolumn numerycznych do dyskretyzacji",
                    size=11,
                    color=AppColors.TEXT_MUTED,
                    italic=True,
                    visible=not has_numeric_columns,
                ),
            ]),

            ft.Container(height=30),
        ]


        if not columns_for_discretization:
            elements.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED,
                               color=AppColors.WARNING, size=40),
                        ft.Container(height=12),
                        ft.Text("Brak kolumn numerycznych możliwych do dyskretyzacji",
                               size=15, color=AppColors.TEXT_PRIMARY,
                               weight=ft.FontWeight.W_600,
                               text_align=ft.TextAlign.CENTER),
                        ft.Container(height=8),
                        ft.Text("Twoje dane zawierają tylko kolumny kategoryczne.\nDyskretyzacja zostanie pominięta.",
                               size=13, color=AppColors.TEXT_SECONDARY,
                               text_align=ft.TextAlign.CENTER),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
                    padding=40,
                    border_radius=12,
                    bgcolor=ft.colors.with_opacity(0.1, AppColors.WARNING),
                    border=ft.border.all(1, AppColors.WARNING),
                )
            )
        else:

            elements.append(
                ft.Column([
                    ft.Text(lang.t('disc_columns_to_discretize'), size=14,
                           color=AppColors.TEXT_SECONDARY,
                           weight=ft.FontWeight.W_600),
                    ft.Container(height=12),


                    ft.Row([
                        ft.OutlinedButton(
                            lang.t('disc_select_all'),
                            icon=ft.icons.CHECK_BOX_ROUNDED,
                            on_click=lambda e: self._select_all_columns(),

                            style=ft.ButtonStyle(
                                color=AppColors.PRIMARY if len(self.selected_columns) < len(columns_for_discretization) else AppColors.TEXT_SECONDARY,
                                side=ft.BorderSide(1, AppColors.PRIMARY if len(self.selected_columns) < len(columns_for_discretization) else AppColors.BORDER),
                            ),
                        ),
                        ft.OutlinedButton(
                            lang.t('disc_deselect_all'),
                            icon=ft.icons.CHECK_BOX_OUTLINE_BLANK_ROUNDED,
                            on_click=lambda e: self._deselect_all_columns(),

                            style=ft.ButtonStyle(
                                color=AppColors.PRIMARY if len(self.selected_columns) > 0 else AppColors.TEXT_SECONDARY,
                                side=ft.BorderSide(1, AppColors.PRIMARY if len(self.selected_columns) > 0 else AppColors.BORDER),
                            ),
                        ),
                    ], spacing=12),

                    ft.Container(height=16),


                    ft.Row([
                        self._create_column_toggle(col)
                        for col in columns_for_discretization
                    ], spacing=8, wrap=True),
                ])
            )

        return ft.Column(elements)

    def _create_column_toggle(self, column_name: str):

        is_selected = column_name in self.selected_columns

        return ft.Container(
            content=ft.Row([
                ft.Icon(
                    ft.icons.CHECK_BOX_ROUNDED if is_selected else ft.icons.CHECK_BOX_OUTLINE_BLANK_ROUNDED,
                    size=18,
                    color=AppColors.PRIMARY if is_selected else AppColors.TEXT_MUTED
                ),
                ft.Text(column_name, size=13,
                       color=AppColors.TEXT_PRIMARY if is_selected else AppColors.TEXT_SECONDARY),
            ], spacing=6),
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            border_radius=8,
            bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if is_selected else AppColors.BG_ELEVATED,
            border=ft.border.all(1, AppColors.PRIMARY if is_selected else AppColors.BORDER),
            on_click=lambda e, col=column_name: self._toggle_column(col),
            ink=True,
        )

    def _build_step_rule_generation(self):



        return ft.Column([
            create_section_header("Generowanie reguł",
                                "Wybierz metodę generowania reguł z danych"),
            ft.Container(height=20),


            ft.Container(
                content=ft.Column([

                    ft.Row([
                        ft.Icon(ft.icons.ACCOUNT_TREE_ROUNDED, color=AppColors.PRIMARY, size=24),
                        ft.Text("Rule Generation", size=16,
                               color=AppColors.TEXT_PRIMARY,
                               weight=ft.FontWeight.W_600),
                    ], spacing=12),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Text("Metoda generowania", size=14,
                               color=AppColors.TEXT_SECONDARY,
                               weight=ft.FontWeight.W_500),
                        ft.Container(height=8),
                        ft.Dropdown(
                            value=self.selected_rule_method,
                            hint_text="Wybierz metodę...",
                            options=[
                                ft.dropdown.Option("Naive", "Naive (1 wiersz = 1 reguła)"),
                                ft.dropdown.Option("Tree", "Tree (Drzewo decyzyjne)"),
                                ft.dropdown.Option("Forest", "Forest (Las losowy - rekomendowane)"),
                            ],
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            text_size=14,
                            on_change=lambda e: self._on_rule_method_change(e),
                        ),
                    ]),
                ], spacing=0),
                padding=24,
                border_radius=12,
                bgcolor=AppColors.BG_ELEVATED,
                border=ft.border.all(1, AppColors.BORDER),
            ),

            ft.Container(height=24),


            self._build_rule_parameters() if self.selected_rule_method else ft.Container(),
        ])

    def _build_rule_parameters(self):

        if self.selected_rule_method == "Naive":

            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.TEXT_MUTED, size=18),
                    ft.Text(
                        "Metoda Naive nie wymaga dodatkowych parametrów. Każdy wiersz staje się regułą.",
                        size=12, color=AppColors.TEXT_MUTED, italic=True),
                ], spacing=12),
                padding=16,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.05, AppColors.TEXT_MUTED),
            )

        elif self.selected_rule_method == "Tree":

            return ft.Container(
                content=ft.Column([

                    ft.Row([
                        ft.Icon(ft.icons.TUNE_ROUNDED, color=AppColors.SECONDARY, size=20),
                        ft.Text("Parametry Drzewa Decyzyjnego", size=14,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=8),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Row([
                            ft.Text("Max Depth (Maksymalna głębokość)", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({int(self.tree_max_depth)})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Container(height=8),
                        ft.TextField(
                            value=str(int(self.tree_max_depth)),
                            keyboard_type=ft.KeyboardType.NUMBER,
                            width=120,
                            height=45,
                            text_size=14,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            hint_text="np. 3",
                            on_change=lambda e: self._on_tree_max_depth_change_textfield(e),
                            error_text=getattr(self, 'tree_max_depth_error', None),
                        ),
                        ft.Text("(większa = bardziej szczegółowe reguły)", size=11,
                               color=AppColors.TEXT_MUTED, italic=True),
                    ]),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Row([
                            ft.Text("Min Samples Leaf (Min. próbek w liściu)", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({int(self.tree_min_samples_leaf)})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Container(height=8),
                        ft.TextField(
                            value=str(int(self.tree_min_samples_leaf)),
                            keyboard_type=ft.KeyboardType.NUMBER,
                            width=120,
                            height=45,
                            text_size=14,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            hint_text="np. 5",
                            on_change=lambda e: self._on_tree_min_samples_leaf_change_textfield(e),
                            error_text=getattr(self, 'tree_min_samples_leaf_error', None),
                        ),
                        ft.Text("(większa = mniej reguł)", size=11,
                               color=AppColors.TEXT_MUTED, italic=True),
                    ]),
                ], spacing=0),
                padding=24,
                border_radius=12,
                bgcolor=AppColors.BG_ELEVATED,
                border=ft.border.all(1, AppColors.BORDER),
            )

        elif self.selected_rule_method == "Forest":

            return ft.Container(
                content=ft.Column([

                    ft.Row([
                        ft.Icon(ft.icons.TUNE_ROUNDED, color=AppColors.SECONDARY, size=20),
                        ft.Text("Parametry Lasu Losowego", size=14,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=8),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Row([
                            ft.Text("Min Depth (Minimalna głębokość)", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({int(self.rf_min_depth)})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Container(height=8),
                        ft.TextField(
                            value=str(int(self.rf_min_depth)),
                            keyboard_type=ft.KeyboardType.NUMBER,
                            width=120,
                            height=45,
                            text_size=14,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            hint_text="1-20",
                            on_change=lambda e: self._on_rf_min_depth_change_textfield(e),
                            error_text=getattr(self, 'rf_min_depth_error', None),
                        ),
                        ft.Text("(1-20, musi być ≤ Max Depth)", size=11,
                               color=AppColors.TEXT_MUTED, italic=True),
                    ]),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Row([
                            ft.Text("Max Depth (Maksymalna głębokość)", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({int(self.rf_max_depth)})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Container(height=8),
                        ft.TextField(
                            value=str(int(self.rf_max_depth)),
                            keyboard_type=ft.KeyboardType.NUMBER,
                            width=120,
                            height=45,
                            text_size=14,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            hint_text="np. 12",
                            on_change=lambda e: self._on_rf_max_depth_change_textfield(e),
                            error_text=getattr(self, 'rf_max_depth_error', None),
                        ),
                        ft.Text("(większa = bardziej szczegółowe reguły)", size=11,
                               color=AppColors.TEXT_MUTED, italic=True),
                    ]),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Row([
                            ft.Text("Min Samples Leaf (Min. próbek w liściu)", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({int(self.rf_min_samples_leaf)})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Container(height=8),
                        ft.TextField(
                            value=str(int(self.rf_min_samples_leaf)),
                            keyboard_type=ft.KeyboardType.NUMBER,
                            width=120,
                            height=45,
                            text_size=14,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            hint_text="np. 5",
                            on_change=lambda e: self._on_rf_min_samples_leaf_change_textfield(e),
                            error_text=getattr(self, 'rf_min_samples_leaf_error', None),
                        ),
                        ft.Text("(większa = mniej reguł)", size=11,
                               color=AppColors.TEXT_MUTED, italic=True),
                    ]),

                    ft.Container(height=16),


                    ft.Column([
                        ft.Row([
                            ft.Text("Number of Estimators (Liczba drzew)", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                        ]),
                        ft.Container(height=8),
                        ft.TextField(
                            value=str(self.rf_n_estimators),
                            keyboard_type=ft.KeyboardType.NUMBER,
                            width=120,
                            height=45,
                            text_size=14,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            hint_text="np. 100",
                            on_change=lambda e: self._on_rf_n_estimators_change(e),
                        ),
                        ft.Text("(więcej = więcej reguł)", size=11,
                               color=AppColors.TEXT_MUTED, italic=True),
                    ]),
                ], spacing=0),
                padding=24,
                border_radius=12,
                bgcolor=AppColors.BG_ELEVATED,
                border=ft.border.all(1, AppColors.BORDER),
            )

        return ft.Container()

    def _on_rule_method_change(self, e):

        self.selected_rule_method = e.control.value
        print(f"[RULE GEN] Wybrano metodę: {self.selected_rule_method}")
        self._update_content()
        self.update()

    def _on_tree_max_depth_change(self, e):

        self.tree_max_depth = int(e.control.value)
        print(f"[RULE GEN] Tree Max Depth: {self.tree_max_depth}")
        self._update_content()
        self.update()

    def _on_tree_min_samples_leaf_change(self, e):

        self.tree_min_samples_leaf = int(e.control.value)
        print(f"[RULE GEN] Tree Min Samples Leaf: {self.tree_min_samples_leaf}")
        self._update_content()
        self.update()

    def _on_rf_n_estimators_change(self, e):

        try:
            value = int(e.control.value) if e.control.value else 1

            if value < 1:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Min. wartość to 1"
                self.rf_n_estimators_error = "Min. wartość to 1"
            elif value > 1000:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Max. wartość to 1000"
                self.rf_n_estimators_error = "Max. wartość to 1000"
            else:
                self.rf_n_estimators = value
                e.control.border_color = AppColors.BORDER
                e.control.error_text = None
                self.rf_n_estimators_error = None
                print(f"[RULE GEN] Forest N Estimators: {self.rf_n_estimators}")
        except ValueError:
            e.control.border_color = ft.colors.RED
            e.control.error_text = "Podaj liczbę całkowitą 1-1000"
            self.rf_n_estimators_error = "Podaj liczbę całkowitą 1-1000"
        e.control.update()



    def _on_tree_max_depth_change_textfield(self, e):

        try:
            value = int(e.control.value) if e.control.value else 1
            if value < 1:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Min. wartość to 1"
                self.tree_max_depth_error = "Min. wartość to 1"
            elif value > 20:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Max. wartość to 20"
                self.tree_max_depth_error = "Max. wartość to 20"
            else:
                self.tree_max_depth = value
                e.control.border_color = AppColors.BORDER
                e.control.error_text = None
                self.tree_max_depth_error = None
                print(f"[RULE GEN] Tree Max Depth: {self.tree_max_depth}")
        except ValueError:
            e.control.border_color = ft.colors.RED
            e.control.error_text = "Podaj liczbę 1-20"
            self.tree_max_depth_error = "Podaj liczbę 1-20"
        e.control.update()

    def _on_tree_min_samples_leaf_change_textfield(self, e):

        try:
            value = int(e.control.value) if e.control.value else 1
            if value < 1:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Min. wartość to 1"
                self.tree_min_samples_leaf_error = "Min. wartość to 1"
            elif value > 100:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Max. wartość to 100"
                self.tree_min_samples_leaf_error = "Max. wartość to 100"
            else:
                self.tree_min_samples_leaf = value
                e.control.border_color = AppColors.BORDER
                e.control.error_text = None
                self.tree_min_samples_leaf_error = None
                print(f"[RULE GEN] Tree Min Samples Leaf: {self.tree_min_samples_leaf}")
        except ValueError:
            e.control.border_color = ft.colors.RED
            e.control.error_text = "Podaj liczbę 1-100"
            self.tree_min_samples_leaf_error = "Podaj liczbę 1-100"
        e.control.update()

    def _on_rf_min_depth_change_textfield(self, e):

        try:
            value = int(e.control.value) if e.control.value else 1
            if value < 1:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Min. wartość to 1"
                self.rf_min_depth_error = "Min. wartość to 1"
            elif value > 20:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Max. wartość to 20"
                self.rf_min_depth_error = "Max. wartość to 20"
            elif value > self.rf_max_depth:
                e.control.border_color = ft.colors.RED
                e.control.error_text = f"Musi być ≤ Max Depth ({int(self.rf_max_depth)})"
                self.rf_min_depth_error = f"Musi być ≤ Max Depth ({int(self.rf_max_depth)})"
            else:
                self.rf_min_depth = value
                e.control.border_color = AppColors.BORDER
                e.control.error_text = None
                self.rf_min_depth_error = None
                print(f"[RULE GEN] Forest Min Depth: {self.rf_min_depth}")
        except ValueError:
            e.control.border_color = ft.colors.RED
            e.control.error_text = "Podaj liczbę 1-20"
            self.rf_min_depth_error = "Podaj liczbę 1-20"
        e.control.update()

    def _on_rf_max_depth_change_textfield(self, e):

        try:
            value = int(e.control.value) if e.control.value else 1
            if value < 1:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Min. wartość to 1"
                self.rf_max_depth_error = "Min. wartość to 1"
            elif value > 20:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Max. wartość to 20"
                self.rf_max_depth_error = "Max. wartość to 20"
            elif value < self.rf_min_depth:
                e.control.border_color = ft.colors.RED
                e.control.error_text = f"Musi być ≥ Min Depth ({int(self.rf_min_depth)})"
                self.rf_max_depth_error = f"Musi być ≥ Min Depth ({int(self.rf_min_depth)})"
            else:
                self.rf_max_depth = value
                e.control.border_color = AppColors.BORDER
                e.control.error_text = None
                self.rf_max_depth_error = None
                print(f"[RULE GEN] Forest Max Depth: {self.rf_max_depth}")
        except ValueError:
            e.control.border_color = ft.colors.RED
            e.control.error_text = "Podaj liczbę 1-20"
            self.rf_max_depth_error = "Podaj liczbę 1-20"
        e.control.update()

    def _on_rf_min_samples_leaf_change_textfield(self, e):

        try:
            value = int(e.control.value) if e.control.value else 1
            if value < 1:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Min. wartość to 1"
                self.rf_min_samples_leaf_error = "Min. wartość to 1"
            elif value > 100:
                e.control.border_color = ft.colors.RED
                e.control.error_text = "Max. wartość to 100"
                self.rf_min_samples_leaf_error = "Max. wartość to 100"
            else:
                self.rf_min_samples_leaf = value
                e.control.border_color = AppColors.BORDER
                e.control.error_text = None
                self.rf_min_samples_leaf_error = None
                print(f"[RULE GEN] Forest Min Samples Leaf: {self.rf_min_samples_leaf}")
        except ValueError:
            e.control.border_color = ft.colors.RED
            e.control.error_text = "Podaj liczbę 1-100"
            self.rf_min_samples_leaf_error = "Podaj liczbę 1-100"
        e.control.update()

    def _create_rule_method_card(self, title: str, subtitle: str, detail: str, icon, enabled: bool = True):




        selected = title == self.selected_rule_method
        is_clickable = enabled

        return ft.Container(
            content=ft.Column([
                ft.Icon(icon, size=32,
                       color=AppColors.PRIMARY if (selected and enabled) else AppColors.TEXT_MUTED),
                ft.Container(height=8),
                ft.Text(title, size=15, color=AppColors.TEXT_PRIMARY if enabled else AppColors.TEXT_MUTED,
                       weight=ft.FontWeight.W_600),
                ft.Text(subtitle, size=12, color=AppColors.TEXT_SECONDARY if enabled else AppColors.TEXT_MUTED),
                ft.Text(detail, size=11, color=AppColors.TEXT_MUTED, italic=True),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
            padding=20,
            border_radius=12,
            bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if selected else AppColors.BG_ELEVATED,
            border=ft.border.all(2, AppColors.PRIMARY if selected else AppColors.BORDER),
            expand=True,
            height=160,
            on_click=lambda e, method=title: self._select_rule_method(method) if is_clickable else None,
            ink=is_clickable,
            opacity=1.0 if enabled else 0.5,
        )

    def _build_rf_configuration(self):




        return ft.Column([
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.TUNE_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text("Konfiguracja Random Forest", size=14,
                           color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                ], spacing=8),
                padding=ft.padding.only(bottom=12),
            ),


            ft.Column([
                ft.Row([
                    ft.Text("Maksymalna głębokość drzewa", size=13,
                           color=AppColors.TEXT_SECONDARY,
                           weight=ft.FontWeight.W_500),
                    ft.Text(f"({int(self.rf_max_depth)})", size=13,
                           color=AppColors.PRIMARY,
                           weight=ft.FontWeight.W_600),
                ]),
                ft.Container(height=8),
                ft.Slider(
                    min=1, max=20, divisions=19, value=self.rf_max_depth,
                    active_color=AppColors.SECONDARY,
                    inactive_color=AppColors.BG_ELEVATED,
                    label="{value}",
                    on_change=lambda e: self._on_rf_max_depth_change(e),
                ),
            ]),

            ft.Container(height=16),


            ft.Column([
                ft.Row([
                    ft.Text("Minimalna liczba próbek do podziału", size=13,
                           color=AppColors.TEXT_SECONDARY,
                           weight=ft.FontWeight.W_500),
                    ft.Text(f"({int(self.rf_min_samples_split)})", size=13,
                           color=AppColors.PRIMARY,
                           weight=ft.FontWeight.W_600),
                ]),
                ft.Container(height=8),
                ft.Slider(
                    min=2, max=20, divisions=18, value=self.rf_min_samples_split,
                    active_color=AppColors.SECONDARY,
                    inactive_color=AppColors.BG_ELEVATED,
                    label="{value}",
                    on_change=lambda e: self._on_rf_min_samples_split_change(e),
                ),
            ]),

            ft.Container(height=16),


            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=18),
                    ft.Column([
                        ft.Text("Parametry Random Forest wpływają na złożoność generowanych reguł.",
                               size=12, color=AppColors.TEXT_SECONDARY),
                        ft.Text("Większa głębokość = bardziej szczegółowe reguły",
                               size=11, color=AppColors.TEXT_MUTED, italic=True),
                    ], spacing=4, expand=True),
                ], spacing=12),
                padding=14,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.05, AppColors.SECONDARY),
                border=ft.border.all(1, ft.colors.with_opacity(0.2, AppColors.SECONDARY)),
            ),
        ])

    def _select_rule_method(self, method: str):

        self.selected_rule_method = method
        print(f"[RULE GEN] Wybrano metodę: {method}")
        self._update_content()
        self.update()

    def _on_rf_min_depth_change(self, e):

        new_min_depth = int(e.control.value)


        if new_min_depth > self.rf_max_depth:

            self.rf_max_depth = new_min_depth
            print(f"[RULE GEN] Min depth zmienione na {new_min_depth}, automatycznie podniesiono max depth do {self.rf_max_depth}")
        else:
            print(f"[RULE GEN] Min depth: {new_min_depth}")

        self.rf_min_depth = new_min_depth
        self._update_content()
        self.update()

    def _on_rf_max_depth_change(self, e):

        new_max_depth = int(e.control.value)


        if new_max_depth < self.rf_min_depth:

            self.rf_min_depth = new_max_depth
            print(f"[RULE GEN] Max depth zmienione na {new_max_depth}, automatycznie obniżono min depth do {self.rf_min_depth}")
        else:
            print(f"[RULE GEN] Max depth: {new_max_depth}")

        self.rf_max_depth = new_max_depth
        self._update_content()
        self.update()

    def _on_rf_min_samples_split_change(self, e):

        self.rf_min_samples_split = int(e.control.value)
        print(f"[RULE GEN] Min samples split: {self.rf_min_samples_split}")
        self._update_content()
        self.update()

    def _build_step_algorithm(self):




        return ft.Column([
            create_section_header("Algorytm wnioskowania",
                                "Wybierz metodę wnioskowania w systemie ekspertowym"),
            ft.Container(height=20),


            ft.Row([
                self._create_algorithm_card(
                    "Forward Chaining",
                    "Wnioskowanie w przód",
                    [
                        "• Data-driven - od faktów do wniosków",
                        "• Opcjonalna optymalizacja (klasteryzacja)",
                        "• Tryb zachłanny (greedy mode)",
                    ],
                    ft.icons.ARROW_FORWARD_ROUNDED,
                    AppColors.PRIMARY
                ),
                self._create_algorithm_card(
                    "Backward Chaining",
                    "Wnioskowanie wstecz",
                    [
                        "• Goal-driven - od hipotezy do faktów",
                        "• Rekurencyjna weryfikacja",
                        "• Wymaga podania celu (atrybut + wartość)",
                    ],
                    ft.icons.ARROW_BACK_ROUNDED,
                    AppColors.SECONDARY
                ),
            ], spacing=16),

            ft.Container(height=24),

          
            self._build_forward_config() if self.selected_algorithm == "Forward Chaining" else ft.Container(),

          
            self._build_backward_config() if self.selected_algorithm == "Backward Chaining" else ft.Container(),
        ])
    
    def _create_algorithm_card(self, title: str, subtitle: str, features: list,
                               icon, color: str):
        selected = title == self.selected_algorithm
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Container(
                        content=ft.Icon(icon, size=24, color=AppColors.TEXT_PRIMARY),
                        padding=12,
                        border_radius=10,
                        bgcolor=color,
                    ),
                    ft.Column([
                        ft.Text(title, size=16, color=AppColors.TEXT_PRIMARY,
                               weight=ft.FontWeight.BOLD),
                        ft.Text(subtitle, size=12, color=AppColors.TEXT_SECONDARY),
                    ], spacing=2),
                ], spacing=12),
                ft.Container(height=12),
                ft.Column([
                    ft.Text(f, size=12, color=AppColors.TEXT_MUTED)
                    for f in features
                ], spacing=4),
            ]),
            padding=20,
            border_radius=12,
            bgcolor=ft.colors.with_opacity(0.1, color) if selected else AppColors.BG_ELEVATED,
            border=ft.border.all(2, color if selected else AppColors.BORDER),
            expand=True,
            on_click=lambda e, alg=title: self._select_algorithm(alg),
            ink=True,
        )

    def _select_algorithm(self, alg: str):

        self.selected_algorithm = alg
        print(f"[ALGORITHM] Wybrano algorytm: {alg}")
        self._update_content()
        self.update()

    def _build_forward_config(self):




        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.TUNE_ROUNDED, color=AppColors.PRIMARY, size=20),
                    ft.Text("Konfiguracja Forward Chaining", size=14,
                           color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                ], spacing=8),

                ft.Container(height=16),


                ft.Row([
                    ft.Switch(
                        value=self.use_clustering,
                        active_color=AppColors.SECONDARY,
                        on_change=lambda e: self._on_clustering_change(e),
                    ),
                    ft.Column([
                        ft.Text("Optymalizacja (Klasteryzacja reguł)", size=13,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ft.Text(
                            "Grupuje reguły w klastry i sprawdza centroidy, aby przyspieszyć wnioskowanie (ClusteredForwardChaining)",
                            size=11, color=AppColors.TEXT_MUTED, italic=True
                        ),
                    ], spacing=4, expand=True),
                ], spacing=12),


                ft.Container(
                    content=ft.Column([
                        ft.Container(height=8),
                        ft.Row([
                            ft.Text("Liczba klastrów", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({int(self.n_clusters)})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Container(height=8),
                        ft.Slider(
                            min=2, max=50, divisions=48, value=self.n_clusters,
                            active_color=AppColors.SECONDARY,
                            inactive_color=AppColors.BG_ELEVATED,
                            label="{value}",
                            on_change=lambda e: self._on_n_clusters_change(e),
                        ),

                        ft.Container(height=12),


                        ft.Text("Metoda obliczania centroidu", size=13,
                               color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                        ft.Container(height=4),
                        ft.Dropdown(
                            value=self.centroid_method,
                            options=[
                                ft.dropdown.Option("general", "General (∩ wspólne) - może być pusty!"),
                                ft.dropdown.Option("specialized", "Specialized (∪ wszystkie) - zalecane"),
                                ft.dropdown.Option("weighted", "Weighted (próg częstości)"),
                            ],
                            width=350,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            on_change=lambda e: self._on_centroid_method_change(e),
                        ),

                        ft.Container(height=12),


                        ft.Row([
                            ft.Text("Próg częstości centroidu", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({self.centroid_threshold:.0%})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Text(
                            "Przesłanka musi występować w >= X% reguł klastra (dla metody Weighted)",
                            size=11, color=AppColors.TEXT_MUTED, italic=True
                        ),
                        ft.Container(height=4),
                        ft.Slider(
                            min=0.1, max=1.0, divisions=9, value=self.centroid_threshold,
                            active_color=AppColors.SECONDARY,
                            inactive_color=AppColors.BG_ELEVATED,
                            label="{value:.0%}",
                            on_change=lambda e: self._on_centroid_threshold_change(e),
                        ),

                        ft.Container(height=12),


                        ft.Row([
                            ft.Text("Min. próg podobieństwa (Alg. 2)", size=13,
                                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                            ft.Text(f"({self.centroid_match_threshold:.0%})", size=13,
                                   color=AppColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ]),
                        ft.Text(
                            "Algorytm 2 (argmax): wybiera klaster z MAX podobieństwem. 0% = akceptuj gdy similarity > 0",
                            size=11, color=AppColors.TEXT_MUTED, italic=True
                        ),
                        ft.Container(height=4),
                        ft.Slider(
                            min=0.0, max=1.0, divisions=10, value=self.centroid_match_threshold,
                            active_color=AppColors.SECONDARY,
                            inactive_color=AppColors.BG_ELEVATED,
                            label="{value:.0%}",
                            on_change=lambda e: self._on_centroid_match_threshold_change(e),
                        ),
                    ]),
                    visible=self.use_clustering,
                ),

                ft.Container(height=12),


                ft.Row([
                    ft.Switch(
                        value=self.use_greedy,
                        active_color=AppColors.SECONDARY,
                        on_change=lambda e: self._on_greedy_change(e),
                    ),
                    ft.Column([
                        ft.Text("Tryb Zachłanny (Greedy)", size=13,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ft.Text(
                            "Aktywuje wszystkie pasujące reguły naraz zamiast pojedynczo (może przyspieszyć wnioskowanie)",
                            size=11, color=AppColors.TEXT_MUTED, italic=True
                        ),
                    ], spacing=4, expand=True),
                ], spacing=12),

                ft.Container(height=12),


                ft.Row([
                    ft.Switch(
                        value=self.use_forward_goal,
                        active_color=AppColors.SECONDARY,
                        on_change=lambda e: self._on_forward_goal_switch_change(e),
                    ),
                    ft.Column([
                        ft.Text("Zatrzymaj po osiągnięciu celu", size=13,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ft.Text(
                            "Przerywa wnioskowanie gdy osiągnięty zostanie określony cel (atrybut, wartość)",
                            size=11, color=AppColors.TEXT_MUTED, italic=True
                        ),
                    ], spacing=4, expand=True),
                ], spacing=12),


                self._build_goal_selection_section(
                    goal_attr=self.forward_goal_attr,
                    goal_value=self.forward_goal_value,
                    on_attr_change=lambda e: self._on_forward_goal_attr_change(e),
                    on_value_change=lambda e: self._on_forward_goal_value_change(e),
                    any_value=self.forward_goal_any_value,
                    on_any_value_change=lambda e: self._on_forward_goal_any_value_change(e),
                    visible=self.use_forward_goal
                ),

                ft.Container(height=16),


                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.PRIMARY, size=18),
                        ft.Text(
                            "Te opcje mogą znacząco przyspieszyć wnioskowanie na dużych zbiorach reguł.",
                            size=12, color=AppColors.TEXT_SECONDARY
                        ),
                    ], spacing=12),
                    padding=14,
                    border_radius=8,
                    bgcolor=ft.colors.with_opacity(0.05, AppColors.PRIMARY),
                    border=ft.border.all(1, ft.colors.with_opacity(0.2, AppColors.PRIMARY)),
                ),
            ]),
            padding=20,
            border_radius=10,
            bgcolor=AppColors.BG_ELEVATED,
        )

    def _build_backward_config(self):




        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.FLAG_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text("Konfiguracja Backward Chaining", size=14,
                           color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                ], spacing=8),

                ft.Container(height=16),




                self._build_goal_selection_section(
                    goal_attr=self.backward_goal_attr,
                    goal_value=self.backward_goal_value,
                    on_attr_change=lambda e: self._on_goal_attr_change(e),
                    on_value_change=lambda e: self._on_goal_value_change(e),
                    any_value=False,
                    on_any_value_change=None,
                    visible=True
                ),

                ft.Container(height=16),


                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=18),
                        ft.Column([
                            ft.Text(
                                "Backward Chaining wymaga podania celu - pary (atrybut, wartość).",
                                size=12, color=AppColors.TEXT_SECONDARY
                            ),
                            ft.Text(
                                "Np. (class, 'setosa') dla Iris dataset.",
                                size=11, color=AppColors.TEXT_MUTED, italic=True
                            ),
                        ], spacing=4, expand=True),
                    ], spacing=12),
                    padding=14,
                    border_radius=8,
                    bgcolor=ft.colors.with_opacity(0.05, AppColors.SECONDARY),
                    border=ft.border.all(1, ft.colors.with_opacity(0.2, AppColors.SECONDARY)),
                ),
            ]),
            padding=20,
            border_radius=10,
            bgcolor=AppColors.BG_ELEVATED,
        )

    def _build_goal_selection_section(self, goal_attr, goal_value, on_attr_change, on_value_change,
                                       any_value=False, on_any_value_change=None, visible=True):















        if not visible:
            return ft.Container()


        available_attrs = []
        if self.discretized_df is not None:
            available_attrs = list(self.discretized_df.columns)


        available_values = []
        if goal_attr and self.discretized_df is not None:
            if goal_attr in self.discretized_df.columns:
                available_values = sorted(self.discretized_df[goal_attr].unique().tolist())


        value_dropdown_disabled = any_value or len(available_values) == 0

        return ft.Column([
            ft.Container(height=12),

            ft.Text("Wybierz cel wnioskowania:", size=13,
                   color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),

            ft.Container(height=12),


            ft.Column([
                ft.Text("Atrybut:", size=12, color=AppColors.TEXT_MUTED),
                ft.Dropdown(
                    options=[ft.dropdown.Option(attr) for attr in available_attrs],
                    value=goal_attr,
                    hint_text="Wybierz atrybut celu",
                    on_change=on_attr_change,
                    border_color=AppColors.BORDER,
                    focused_border_color=AppColors.SECONDARY,
                ),
            ], spacing=4),

            ft.Container(height=12),


            ft.Checkbox(
                label="Dowolna wartość (zatrzymaj gdy znajdzie jakąkolwiek wartość tego atrybutu)",
                value=any_value,
                on_change=on_any_value_change,
                active_color=AppColors.SECONDARY,
            ) if on_any_value_change else ft.Container(),

            ft.Container(height=8),


            ft.Column([
                ft.Text("Wartość:" + (" (wyłączone - dowolna wartość)" if any_value else ""),
                       size=12, color=AppColors.TEXT_MUTED),
                ft.Dropdown(
                    options=[ft.dropdown.Option(val) for val in available_values],
                    value=goal_value if not any_value else None,
                    hint_text="Dowolna wartość" if any_value else (
                        "Wybierz wartość celu" if available_values else "Najpierw wybierz atrybut"),
                    on_change=on_value_change,
                    border_color=AppColors.BORDER,
                    focused_border_color=AppColors.SECONDARY,
                    disabled=value_dropdown_disabled,
                ),
            ], spacing=4),
        ], spacing=0)

    def _on_clustering_change(self, e):

        self.use_clustering = e.control.value
        print(f"[ALGORITHM] Optymalizacja (clustering): {self.use_clustering}")

        self._update_content()
        self.update()

    def _on_n_clusters_change(self, e):

        self.n_clusters = int(e.control.value)
        print(f"[ALGORITHM] Liczba klastrów: {self.n_clusters}")
        self._update_content()
        self.update()

    def _on_centroid_method_change(self, e):

        self.centroid_method = e.control.value
        print(f"[ALGORITHM] Metoda centroidu: {self.centroid_method}")
        self._update_content()
        self.update()

    def _on_centroid_threshold_change(self, e):

        self.centroid_threshold = float(e.control.value)
        print(f"[ALGORITHM] Próg częstości centroidu: {self.centroid_threshold:.0%}")
        self._update_content()
        self.update()

    def _on_centroid_match_threshold_change(self, e):

        self.centroid_match_threshold = float(e.control.value)
        print(f"[ALGORITHM] Próg dopasowania centroidu: {self.centroid_match_threshold:.0%}")
        self._update_content()
        self.update()

    def _on_greedy_change(self, e):

        self.use_greedy = e.control.value
        print(f"[ALGORITHM] Tryb zachłanny: {self.use_greedy}")
        self.update()

    def _on_goal_attr_change(self, e):

        self.backward_goal_attr = e.control.value
        self.backward_goal_value = None
        print(f"[ALGORITHM] Cel - atrybut: {self.backward_goal_attr}")
        self._update_content()
        self.update()

    def _on_goal_value_change(self, e):

        self.backward_goal_value = e.control.value
        print(f"[ALGORITHM] Cel - wartość: {self.backward_goal_value}")
        self.update()

    def _on_backward_goal_any_value_change(self, e):

        self.backward_goal_any_value = e.control.value
        if self.backward_goal_any_value:
            self.backward_goal_value = None
        print(f"[ALGORITHM] Backward - dowolna wartość: {self.backward_goal_any_value}")
        self._update_content()
        self.update()

    def _on_forward_goal_switch_change(self, e):

        self.use_forward_goal = e.control.value
        print(f"[ALGORITHM] Forward - użyj celu: {self.use_forward_goal}")
        self._update_content()
        self.update()

    def _on_forward_goal_attr_change(self, e):

        self.forward_goal_attr = e.control.value
        self.forward_goal_value = None
        print(f"[ALGORITHM] Forward - cel atrybut: {self.forward_goal_attr}")
        self._update_content()
        self.update()

    def _on_forward_goal_value_change(self, e):

        self.forward_goal_value = e.control.value
        print(f"[ALGORITHM] Forward - cel wartość: {self.forward_goal_value}")
        self.update()

    def _on_forward_goal_any_value_change(self, e):

        self.forward_goal_any_value = e.control.value
        if self.forward_goal_any_value:
            self.forward_goal_value = None
        print(f"[ALGORITHM] Forward - dowolna wartość: {self.forward_goal_any_value}")
        self._update_content()
        self.update()

    def _build_step_strategy(self):
        return ft.Column([
            create_section_header("Strategia rozwiązywania konfliktów",
                                "Jak wybierać regułę gdy wiele pasuje (conflict set)?"),
            ft.Container(height=10),


            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=18),
                    ft.Text(
                        "Tryb Benchmark: Zaznacz wiele strategii do porównania. Tryb Single Run: Użyta zostanie pierwsza zaznaczona.",
                        size=12, color=AppColors.TEXT_SECONDARY
                    ),
                ], spacing=10),
                padding=12,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.1, AppColors.SECONDARY),
            ),

            ft.Container(height=16),

            ft.Row([
                self._create_strategy_card(
                    "Random", "Losowy wybór",
                    "Niedeterministyczna", ft.icons.CASINO_ROUNDED),
                self._create_strategy_card(
                    "Textual Order", "Pierwsza na liście",
                    "FIFO - kolejność definicji", ft.icons.FORMAT_LIST_NUMBERED_ROUNDED),
                self._create_strategy_card(
                    "Recency", "Najnowsze fakty",
                    "LIFO - ostatnio dodane", ft.icons.UPDATE_ROUNDED),
                self._create_strategy_card(
                    "Specificity", "Najwięcej przesłanek",
                    "Bardziej szczegółowe", ft.icons.TUNE_ROUNDED),
            ], spacing=12),

            ft.Container(height=16),


            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_ROUNDED,
                           color=AppColors.SECONDARY if self.selected_strategies else AppColors.TEXT_MUTED,
                           size=20),
                    ft.Text(
                        f"Wybrano: {', '.join(sorted(self.selected_strategies)) if self.selected_strategies else 'Brak'}" ,
                        size=13,
                        color=AppColors.TEXT_PRIMARY if self.selected_strategies else AppColors.TEXT_MUTED,
                        weight=ft.FontWeight.W_500 if self.selected_strategies else None
                    ),
                ], spacing=12),
                padding=16,
                border_radius=8,
                bgcolor=AppColors.BG_ELEVATED,
            ),

            ft.Container(height=12),

            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.PRIMARY, size=20),
                    ft.Text(
                        "Reguła raz użyta nie może być użyta ponownie w tym samym wnioskowania.",
                        size=13, color=AppColors.TEXT_SECONDARY
                    ),
                ], spacing=12),
                padding=16,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY),
            ),
        ])
    
    def _create_strategy_card(self, title: str, subtitle: str, detail: str, icon):

        selected = title in self.selected_strategies
        return ft.Container(
            content=ft.Column([

                ft.Row([
                    ft.Checkbox(
                        value=selected,
                        on_change=lambda e, strategy=title: self._toggle_strategy(strategy),
                        active_color=AppColors.PRIMARY,
                        check_color=AppColors.TEXT_PRIMARY,
                    ),
                ], alignment=ft.MainAxisAlignment.END),
                ft.Icon(icon, size=28,
                       color=AppColors.PRIMARY if selected else AppColors.TEXT_MUTED),
                ft.Container(height=4),
                ft.Text(title, size=14, color=AppColors.TEXT_PRIMARY,
                       weight=ft.FontWeight.W_600),
                ft.Text(subtitle, size=11, color=AppColors.TEXT_SECONDARY),
                ft.Text(detail, size=10, color=AppColors.TEXT_MUTED, italic=True),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
            padding=ft.padding.only(left=16, right=16, top=4, bottom=16),
            border_radius=12,
            bgcolor=ft.colors.with_opacity(0.15, AppColors.PRIMARY) if selected else AppColors.BG_ELEVATED,
            border=ft.border.all(2, AppColors.PRIMARY if selected else AppColors.BORDER),
            expand=True,
            height=160,
            on_click=lambda e, strategy=title: self._toggle_strategy(strategy),
            ink=True,
        )
    
    def _validate_facts_percent(self, value: str) -> tuple[bool, str, list]:




        try:

            parts = [p.strip() for p in value.split(',') if p.strip()]

            if not parts:
                return False, "Pole nie może być puste", []


            numbers = []
            for part in parts:
                num = int(part)
                if num < 1 or num > 100:
                    return False, f"Wartość {num} poza zakresem 1-100", []
                numbers.append(num)

            return True, "", numbers
        except ValueError:
            return False, "Błąd: liczby muszą być całkowite (1-100), oddzielone przecinkami", []

    def _validate_repetitions(self, value: str) -> tuple[bool, str, int]:




        try:
            num = int(value.strip())
            if num <= 0:
                return False, "Wartość musi być większa od 0", 0
            return True, "", num
        except ValueError:
            return False, "Błąd: wartość musi być liczbą całkowitą > 0", 0

    def _on_facts_change(self, e):

        self.initial_facts_percent = e.control.value
        is_valid, error_msg, values = self._validate_facts_percent(e.control.value)

        if is_valid:
            self.facts_validation_error = None
            e.control.border_color = AppColors.BORDER
            e.control.error_text = None
        else:
            self.facts_validation_error = error_msg
            e.control.border_color = ft.colors.RED
            e.control.error_text = error_msg


        e.control.update()

    def _on_repetitions_change(self, e):

        self.repetitions = e.control.value
        is_valid, error_msg, value = self._validate_repetitions(e.control.value)

        if is_valid:
            self.repetitions_validation_error = None
            e.control.border_color = AppColors.BORDER
            e.control.error_text = None
        else:
            self.repetitions_validation_error = error_msg
            e.control.border_color = ft.colors.RED
            e.control.error_text = error_msg


        e.control.update()

    def _build_step_run(self):





        rules_count = len(self.generated_rules) if self.generated_rules else 0
        rules_text = f"{rules_count} reguł"


        algorithm_name = self.inference_config.get("method", "forward")
        if algorithm_name == "forward":
            algorithm_text = "Forward Chaining"
        else:
            algorithm_text = "Backward Chaining"


        optimizations = []
        if self.inference_config.get("use_clustering", False):
            optimizations.append("Klasteryzacja")
        if self.inference_config.get("use_greedy", False):
            optimizations.append("Greedy")
        optimization_text = ", ".join(optimizations) if optimizations else "Brak"


        goal = self.inference_config.get("goal", None)
        if goal:
            goal_text = f"{goal[0]} = {goal[1]}"
        else:
            goal_text = "Brak (wnioskowanie pełne)"


        if self.selected_strategies:
            strategy_text = ", ".join(sorted(self.selected_strategies))
        else:
            strategy_text = self.selected_strategy or "Brak"


        facts_valid, _, facts_values = self._validate_facts_percent(self.initial_facts_percent)
        reps_valid, _, reps_value = self._validate_repetitions(self.repetitions)
        num_strategies = len(self.selected_strategies)
        if facts_valid and reps_valid and num_strategies > 0:
            total_benchmark_runs = len(facts_values) * num_strategies * reps_value
            benchmark_info = f"{len(facts_values)} % × {num_strategies} strategii × {reps_value} powtórzeń = {total_benchmark_runs} uruchomień"
        else:
            total_benchmark_runs = 0
            benchmark_info = "Uzupełnij konfigurację"

        return ft.Column([
            create_section_header("Uruchomienie wnioskowania",
                                "Podsumowanie konfiguracji"),
            ft.Container(height=16),


            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.icons.SETTINGS_ROUNDED, color=AppColors.PRIMARY, size=24),
                        ft.Text("Podsumowanie konfiguracji", size=16,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=12),

                    ft.Container(height=12),

                    self._create_summary_row("Liczba reguł", rules_text),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_summary_row("Algorytm", algorithm_text),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_summary_row("Optymalizacje", optimization_text),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_summary_row("Strategia", strategy_text),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_summary_row("Cel", goal_text),
                ], spacing=10),
                padding=16,
                border_radius=10,
                bgcolor=AppColors.BG_ELEVATED,
            ),

            ft.Container(height=20),




            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.icons.SCIENCE_ROUNDED, color=AppColors.SECONDARY, size=24),
                        ft.Text("Tryb Benchmark (badawczy)", size=16,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                    ], spacing=12),

                    ft.Container(height=12),


                    ft.Row([
                        ft.Text("% Wiedzy:", size=13, color=AppColors.TEXT_SECONDARY, width=100),
                        ft.TextField(
                            value=self.initial_facts_percent,
                            hint_text="np. 10, 25, 50, 100",
                            on_change=self._on_facts_change,
                            width=200,
                            height=40,
                            text_size=13,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            error_text=self.facts_validation_error if hasattr(self, 'facts_validation_error') else None,
                        ),
                        ft.Text("(procent faktów znanych na starcie)", size=11, color=AppColors.TEXT_MUTED, italic=True),
                    ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),

                    ft.Container(height=8),


                    ft.Row([
                        ft.Text("Powtórzeń (N):", size=13, color=AppColors.TEXT_SECONDARY, width=100),
                        ft.TextField(
                            value=self.repetitions,
                            hint_text="np. 100",
                            on_change=self._on_repetitions_change,
                            width=100,
                            height=40,
                            text_size=13,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            error_text=self.repetitions_validation_error if hasattr(self, 'repetitions_validation_error') else None,
                        ),
                        ft.Text("(liczba powtórzeń dla każdej konfiguracji)", size=11, color=AppColors.TEXT_MUTED, italic=True),
                    ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),

                    ft.Container(height=12),


                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.icons.CALCULATE_ROUNDED,
                                   color=AppColors.SECONDARY if total_benchmark_runs > 0 else AppColors.TEXT_MUTED,
                                   size=18),
                            ft.Text(benchmark_info, size=12,
                                   color=AppColors.TEXT_PRIMARY if total_benchmark_runs > 0 else AppColors.TEXT_MUTED),
                        ], spacing=8),
                        padding=12,
                        border_radius=8,
                        bgcolor=ft.colors.with_opacity(0.1, AppColors.SECONDARY) if total_benchmark_runs > 0 else AppColors.BG_CARD,
                    ),

                    ft.Container(height=16),


                    ft.ElevatedButton(
                        text="Uruchom Benchmark",
                        icon=ft.icons.SCIENCE_ROUNDED,
                        bgcolor=AppColors.SECONDARY,
                        color=AppColors.TEXT_PRIMARY,
                        width=250,
                        height=45,
                        disabled=total_benchmark_runs == 0,
                        on_click=lambda e: self._execute_benchmark(),
                    ),

                ], spacing=8),
                padding=20,
                border_radius=10,
                bgcolor=AppColors.BG_CARD,
                border=ft.border.all(1, AppColors.SECONDARY),
            ),

            ft.Container(height=16),


            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=AppColors.PRIMARY, size=18),
                    ft.Text(
                        f"Przycisk 'Uruchom eksperyment' (na dole) uruchamia jedno wnioskowanie dla każdej wybranej strategii ({len(self.selected_strategies)} strategii).",
                        size=12, color=AppColors.TEXT_MUTED
                    ),
                ], spacing=10),
                padding=12,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.05, AppColors.PRIMARY),
            ),

            ft.Container(height=16),


            self.inference_results_container if hasattr(self, 'inference_results_container') else ft.Container(),
        ])
    
    def _create_experiments_count_info(self):


        facts_valid, _, facts_values = self._validate_facts_percent(self.initial_facts_percent)
        reps_valid, _, reps_value = self._validate_repetitions(self.repetitions)

        if facts_valid and reps_valid:
            num_facts = len(facts_values)
            total_experiments = num_facts * reps_value
            info_text = f"Zostanie uruchomionych łącznie {total_experiments} eksperymentów ({num_facts} wartości procentowych × {reps_value} powtórzeń)"
            text_color = AppColors.TEXT_SECONDARY
        else:
            info_text = "Nieprawidłowe dane"
            text_color = ft.colors.RED

        return ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED,
                       color=AppColors.PRIMARY if facts_valid and reps_valid else ft.colors.RED,
                       size=18),
                ft.Text(info_text, size=12, color=text_color, italic=True),
            ], spacing=8),
            padding=12,
            border_radius=8,
            bgcolor=ft.colors.with_opacity(0.1, AppColors.PRIMARY) if facts_valid and reps_valid else ft.colors.with_opacity(0.1, ft.colors.RED),
        )

    def _create_summary_row(self, label: str, value: str):
        return ft.Row([
            ft.Text(label, size=13, color=AppColors.TEXT_MUTED, width=120),
            ft.Text(value, size=13, color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
        ])

    def _execute_inference(self):
















        print("\n" + "="*60)
        print("BACKEND-05: URUCHOMIENIE WNIOSKOWANIA (ExperimentRunner)")
        print("="*60)

        try:





            if self.loaded_df is None or self.loaded_df.empty:
                print("[ERROR] Brak wczytanych danych!")
                self._show_snackbar("Błąd: Brak danych do wnioskowania", success=False)
                return


            if not self.csv_decision_column or self.csv_decision_column not in self.loaded_df.columns:
                print("[ERROR] Brak kolumny decyzyjnej!")
                self._show_snackbar("Błąd: Brak kolumny decyzyjnej", success=False)
                return


            if not self.selected_strategies:
                print("[ERROR] Brak wybranych strategii!")
                self._show_snackbar("Błąd: Wybierz co najmniej jedną strategię", success=False)
                return

            print(f"[INFERENCE] Użycie ExperimentRunner na DataFrame: {self.loaded_df.shape}")
            print(f"[INFERENCE] Wybrane strategie: {self.selected_strategies}")






            strategy_map = {
                "Random": InferenceStrategy.RANDOM,
                "Textual Order": InferenceStrategy.FIRST,
                "Specificity": InferenceStrategy.SPECIFICITY,
                "Recency": InferenceStrategy.RECENCY
            }


            discretization_method_map = {
                "Equal Width": "equal_width",
                "Equal Frequency": "equal_frequency",
                "K-Means": "kmeans",
                "Brak": "equal_width"
            }
            discretization_method = discretization_method_map.get(
                self.selected_discretization,
                "equal_width"
            )

            rule_method_map = {
                "Naive": RuleGenerationMethod.NAIVE,
                "Tree": RuleGenerationMethod.TREE,
                "Forest": RuleGenerationMethod.FOREST,
            }
            generate_method = rule_method_map.get(self.selected_rule_method, RuleGenerationMethod.FOREST)


            method = self.inference_config.get("method", "forward")
            use_greedy = self.inference_config.get("use_greedy", False)

            if method == "backward":
                inference_method = InferenceMethod.BACKWARD
            elif use_greedy:
                inference_method = InferenceMethod.GREEDY
            else:
                inference_method = InferenceMethod.FORWARD


            goal = self.inference_config.get("goal", None)


            use_clustering = self.inference_config.get("use_clustering", False)

            seed = int(self.random_seed) if self.random_seed else 42





            all_results = []
            strategies_list = sorted(list(self.selected_strategies))
            total_strategies = len(strategies_list)

            print(f"\n[INFERENCE] Uruchamiam wnioskowanie dla {total_strategies} strategii...")

            for idx, strategy_name in enumerate(strategies_list, 1):
                strategy = strategy_map.get(strategy_name)
                if not strategy:
                    print(f"[WARNING] Nieznana strategia: {strategy_name}, pomijam")
                    continue

                print(f"\n[INFERENCE] [{idx}/{total_strategies}] Strategia: {strategy_name}")


                config = ExperimentConfig(
                    seed=seed,
                    strategy=strategy,
                    generate_method=generate_method,
                    inference_method=inference_method,
                    decision_column=self.csv_decision_column,


                    discretization_method=discretization_method,
                    discretization_bins=int(self.n_bins) if hasattr(self, 'n_bins') else 3,

                    tree_max_depth=int(self.tree_max_depth),
                    tree_min_samples_leaf=int(self.tree_min_samples_leaf),

                    forest_n_estimators=int(self.rf_n_estimators),
                    forest_min_depth=int(self.rf_min_depth),
                    forest_max_depth=int(self.rf_max_depth),
                    forest_min_samples_leaf=int(self.rf_min_samples_leaf),

                    clustering_enabled=use_clustering,
                    n_clusters=int(self.n_clusters),
                    centroid_method=self.centroid_method,
                    centroid_threshold=float(self.centroid_threshold),
                    centroid_match_threshold=float(self.centroid_match_threshold),


                    goal=goal,


                    skip_validation=self.skip_validation
                )

                print(f"[INFERENCE] ========== URUCHOMIENIE ExperimentRunner ({strategy_name}) ==========")
                runner = ExperimentRunner(config)


                result = runner.run(self.loaded_df)

                print(f"[INFERENCE] [{strategy_name}] Sukces: {result.success}")
                print(f"[INFERENCE] [{strategy_name}] Czas: {result.execution_time_ms:.2f} ms")
                print(f"[INFERENCE] [{strategy_name}] Nowych faktów: {len(result.new_facts)}")

                all_results.append({
                    'strategy': strategy_name,
                    'result': result,
                    'runner': runner
                })

            print(f"\n[INFERENCE] ========== WSZYSTKIE WNIOSKOWANIA ZAKOŃCZONE ==========")
            print(f"[INFERENCE] Wykonano {len(all_results)} uruchomień")






            if all_results:
                last_result = all_results[-1]
                self._display_inference_results(last_result['result'], last_result['result'].execution_time_ms)


                total_facts = sum(len(r['result'].new_facts) for r in all_results)
                total_time = sum(r['result'].execution_time_ms for r in all_results)
                self._show_snackbar(
                    f"Zakończono {len(all_results)} wnioskowań: {total_facts} nowych faktów w {total_time:.0f} ms",
                    success=True
                )






            def navigate_to_results_after_delay():
                import time
                time.sleep(1.5)


                async def do_navigate():
                    if self.page and hasattr(self.page, 'sidebar'):

                        self.page.sidebar._on_click(2)

                if self.page:
                    self.page.run_task(do_navigate)


            threading.Thread(target=navigate_to_results_after_delay, daemon=True).start()

        except Exception as e:
            print(f"[ERROR] Błąd podczas wnioskowania: {e}")
            import traceback
            traceback.print_exc()


            self.next_button.disabled = False
            self.next_button.text = "Spróbuj ponownie"
            self.next_button.icon = ft.icons.REFRESH_ROUNDED
            self.next_button.bgcolor = AppColors.PRIMARY
            self.update()


            self._show_snackbar(f"Błąd: {str(e)}", success=False)





    def _execute_benchmark(self):











        print("\n" + "="*70)
        print("BENCHMARK: URUCHOMIENIE TRYBU BADAWCZEGO")
        print("="*70)





        if self.loaded_df is None or self.loaded_df.empty:
            self._show_snackbar("Błąd: Brak wczytanych danych", success=False)
            return

        if not self.csv_decision_column:
            self._show_snackbar("Błąd: Brak kolumny decyzyjnej", success=False)
            return

        if not self.selected_strategies:
            self._show_snackbar("Błąd: Wybierz co najmniej jedną strategię", success=False)
            return


        facts_valid, facts_error, knowledge_percentages = self._validate_facts_percent(self.initial_facts_percent)
        if not facts_valid:
            self._show_snackbar(f"Błąd: {facts_error}", success=False)
            return

        reps_valid, reps_error, n_repetitions = self._validate_repetitions(self.repetitions)
        if not reps_valid:
            self._show_snackbar(f"Błąd: {reps_error}", success=False)
            return

        strategies = sorted(list(self.selected_strategies))
        total_runs = len(knowledge_percentages) * len(strategies) * n_repetitions

        print(f"[BENCHMARK] Knowledge %: {knowledge_percentages}")
        print(f"[BENCHMARK] Strategies: {strategies}")
        print(f"[BENCHMARK] Repetitions: {n_repetitions}")
        print(f"[BENCHMARK] Total runs: {total_runs}")






        self.next_button.disabled = True
        self.next_button.text = "Benchmark w toku..."
        self.next_button.icon = ft.icons.HOURGLASS_EMPTY_ROUNDED


        self.benchmark_progress_bar = ft.ProgressBar(value=0, width=400, color=AppColors.SECONDARY)
        self.benchmark_status_text = ft.Text("Inicjalizacja...", size=13, color=AppColors.TEXT_SECONDARY)

        progress_container = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.SCIENCE_ROUNDED, color=AppColors.SECONDARY, size=24),
                    ft.Text("Benchmark w toku", size=16, color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                ], spacing=12),
                ft.Container(height=12),
                self.benchmark_progress_bar,
                ft.Container(height=8),
                self.benchmark_status_text,
            ]),
            padding=20,
            border_radius=10,
            bgcolor=AppColors.BG_ELEVATED,
        )


        self.inference_results_container.content = progress_container
        self.update()





        def run_benchmark_thread():
            try:

                seed = int(self.random_seed) if self.random_seed else 42
                random.seed(seed)


                df = self.loaded_df.copy()
                decision_column = self.csv_decision_column





                self.benchmark_status_text.value = "Preprocessing danych..."
                if self.page:
                    self.page.update()


                discretization_method_map = {
                    "Equal Width": "equal_width",
                    "Equal Frequency": "equal_frequency",
                    "K-Means": "kmeans",
                    "Brak": "equal_width"
                }
                discretization_method = discretization_method_map.get(self.selected_discretization, "equal_width")


                rule_method_map = {
                    "Naive": RuleGenerationMethod.NAIVE,
                    "Tree": RuleGenerationMethod.TREE,
                    "Forest": RuleGenerationMethod.FOREST,
                }
                generate_method = rule_method_map.get(self.selected_rule_method, RuleGenerationMethod.FOREST)


                base_config = ExperimentConfig(
                    seed=seed,
                    strategy=InferenceStrategy.RANDOM,
                    generate_method=generate_method,
                    inference_method=InferenceMethod.FORWARD,
                    decision_column=decision_column,
                    discretization_method=discretization_method,
                    discretization_bins=int(self.n_bins) if hasattr(self, 'n_bins') else 3,
                    tree_max_depth=int(self.tree_max_depth),
                    tree_min_samples_leaf=int(self.tree_min_samples_leaf),
                    forest_n_estimators=int(self.rf_n_estimators),
                    forest_min_depth=int(self.rf_min_depth),
                    forest_max_depth=int(self.rf_max_depth),
                    forest_min_samples_leaf=int(self.rf_min_samples_leaf),
                    clustering_enabled=False,
                    centroid_method=self.centroid_method,
                    centroid_threshold=float(self.centroid_threshold),
                    centroid_match_threshold=float(self.centroid_match_threshold),
                    skip_validation=True,
                )


                runner = ExperimentRunner(base_config, enable_storage=False)
                runner._set_global_seed()
                runner._discretize_data(df)
                runner._generate_rules()
                generated_rules = runner.generated_rules
                discretized_df = runner.discretized_df

                print(f"[BENCHMARK] Wygenerowano {len(generated_rules)} reguł")
                self.benchmark_status_text.value = f"Wygenerowano {len(generated_rules)} reguł. Rozpoczynam benchmark..."
                if self.page:
                    self.page.update()





                results = []
                run_counter = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


                strategy_class_map = {
                    "Random": RandomStrategy,
                    "Textual Order": FirstStrategy,
                    "Recency": RecencyStrategy,
                    "Specificity": SpecificityStrategy,
                }

                for percent in knowledge_percentages:
                    for strategy_name in strategies:
                        for rep in range(n_repetitions):
                            run_counter += 1


                            progress = run_counter / total_runs
                            self.benchmark_progress_bar.value = progress
                            self.benchmark_status_text.value = f"Run {run_counter}/{total_runs}: {strategy_name}, {percent}% wiedzy"
                            if self.page and run_counter % 5 == 0:
                                self.page.update()


                            test_row_idx = random.randint(0, len(discretized_df) - 1)
                            test_row = discretized_df.iloc[test_row_idx]


                            true_decision = str(test_row[decision_column])


                            all_facts = []
                            for column, value in test_row.items():
                                if column == decision_column:
                                    continue
                                if pd.isna(value):
                                    continue
                                all_facts.append(Fact(attribute=column, value=str(value)))


                            if percent >= 100:
                                selected_facts = set(all_facts)
                            else:
                                k = max(1, int(len(all_facts) * percent / 100))
                                selected_facts = set(random.sample(all_facts, min(k, len(all_facts))))




                            start_time = time.perf_counter()

                            kb = KnowledgeBase(rules=generated_rules, facts=selected_facts)
                            strategy_class = strategy_class_map.get(strategy_name, RandomStrategy)
                            strategy = strategy_class()
                            engine = ForwardChaining(strategy=strategy)
                            result = engine.run(kb, goal=decision_column)

                            elapsed_ms = (time.perf_counter() - start_time) * 1000


                            predicted_decision = None
                            for fact in result.facts:
                                if fact.attribute == decision_column:
                                    predicted_decision = fact.value
                                    break

                            is_correct = (predicted_decision == true_decision)


                            avg_conflict_size = result.rules_activated / result.iterations if result.iterations > 0 else 0
                            results.append({
                                "RunID": run_counter,
                                "Strategy": strategy_name,
                                "Knowledge_Percent": percent,
                                "Time_ms": round(elapsed_ms, 3),
                                "Rules_Fired": len(result.rules_fired),
                                "Correctness": is_correct,
                                "Predicted": predicted_decision or "None",
                                "Actual": true_decision,
                                "Conflict_Set_Avg": round(avg_conflict_size, 2),
                                "Iterations": result.iterations,
                                "Winning_Rule_ID": result.rules_fired[0].id if result.rules_fired else "None",
                                "_trace": result.trace,
                                "_strategy_name": strategy_name,
                                "_percent": percent,
                            })





                self.benchmark_status_text.value = "Zapisywanie wyników..."
                if self.page:
                    self.page.update()


                results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                os.makedirs(results_dir, exist_ok=True)

                benchmark_folder = os.path.join(results_dir, f"benchmark_{timestamp}")
                os.makedirs(benchmark_folder, exist_ok=True)


                traces_folder = os.path.join(benchmark_folder, "traces")
                os.makedirs(traces_folder, exist_ok=True)


                self.benchmark_status_text.value = "Zapisywanie trace files..."
                if self.page:
                    self.page.update()

                for result_data in results:
                    run_id = result_data["RunID"]
                    strategy = result_data["_strategy_name"]
                    percent = result_data["_percent"]
                    trace_content = result_data["_trace"]


                    trace_filename = f"run_{run_id}_{strategy}_{percent}pct.txt"
                    trace_path = os.path.join(traces_folder, trace_filename)

                    with open(trace_path, 'w', encoding='utf-8') as f:
                        f.write(f"=== DEEP INVESTIGATION TRACE ===\n")
                        f.write(f"Run ID: {run_id}\n")
                        f.write(f"Strategy: {strategy}\n")
                        f.write(f"Knowledge Percent: {percent}%\n")
                        f.write(f"Result: {'CORRECT' if result_data['Correctness'] else 'INCORRECT'}\n")
                        f.write(f"Predicted: {result_data['Predicted']}, Actual: {result_data['Actual']}\n")
                        f.write("\n")
                        f.write("\n".join(trace_content))

                print(f"[BENCHMARK] Zapisano {len(results)} trace files do: {traces_folder}")


                csv_results = []
                for r in results:
                    csv_row = {k: v for k, v in r.items() if not k.startswith('_')}
                    csv_results.append(csv_row)

                csv_path = os.path.join(benchmark_folder, "results.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_results)


                config_data = {
                    "timestamp": datetime.now().isoformat(),
                    "seed": seed,
                    "knowledge_percentages": knowledge_percentages,
                    "strategies": strategies,
                    "repetitions_per_config": n_repetitions,
                    "total_runs": total_runs,
                    "rule_generation": {
                        "method": self.selected_rule_method,
                        "min_depth": int(self.rf_min_depth),
                        "max_depth": int(self.rf_max_depth),
                        "n_estimators": int(self.rf_n_estimators),
                    },
                    "discretization": {
                        "method": self.selected_discretization,
                        "bins": int(self.n_bins) if hasattr(self, 'n_bins') else 3,
                    },
                    "dataset": self.loaded_file_path or "unknown",
                    "decision_column": decision_column,
                    "rules_count": len(generated_rules),
                }
                json_path = os.path.join(benchmark_folder, "config.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)


                rules_path = os.path.join(benchmark_folder, "rules_dump.txt")
                with open(rules_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== GENERATED RULES ({len(generated_rules)} total) ===\n\n")
                    for i, rule in enumerate(generated_rules, 1):
                        premises_str = " AND ".join([f"{p.attribute}={p.value}" for p in rule.premises])
                        conclusion_str = f"{rule.conclusion.attribute}={rule.conclusion.value}"
                        f.write(f"Rule {i}: IF ({premises_str}) THEN {conclusion_str}\n")


                raw_csv_path = os.path.join(benchmark_folder, "dataset_raw.csv")
                df.to_csv(raw_csv_path, index=False, encoding='utf-8')
                print(f"[BENCHMARK] Saved dataset_raw.csv ({len(df)} rows)")


                processed_csv_path = os.path.join(benchmark_folder, "dataset_processed.csv")
                discretized_df.to_csv(processed_csv_path, index=False, encoding='utf-8')
                print(f"[BENCHMARK] Saved dataset_processed.csv ({len(discretized_df)} rows)")


                rule_gen_log_path = os.path.join(benchmark_folder, "rule_generation_log.txt")
                with open(rule_gen_log_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 70 + "\n")
                    f.write("RULES GENERATION LOG - FULL TRANSPARENCY\n")
                    f.write("=" * 70 + "\n\n")


                    f.write(f"Generation method: {self.selected_rule_method}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Decision column: {decision_column}\n")
                    f.write(f"Number of rows in dataset: {len(df)}\n")
                    f.write(f"Number of columns (features): {len(df.columns) - 1}\n")
                    f.write(f"Columns: {list(df.columns)}\n\n")


                    f.write("-" * 50 + "\n")
                    f.write("DISCRETIZATION\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Method: {self.selected_discretization}\n")
                    f.write(f"Number of bins: {int(self.n_bins) if hasattr(self, 'n_bins') else 3}\n\n")

                    if self.selected_rule_method == "Tree":

                        f.write("-" * 50 + "\n")
                        f.write("DECISION TREE PARAMETERS\n")
                        f.write("-" * 50 + "\n")
                        f.write(f"max_depth: {int(self.tree_max_depth)}\n")
                        f.write(f"min_samples_leaf: {int(self.tree_min_samples_leaf)}\n")
                        f.write(f"random_state: {seed}\n\n")

                        f.write("-" * 50 + "\n")
                        f.write("GENERATION PROCESS\n")
                        f.write("-" * 50 + "\n")
                        f.write("1. Encoded categorical data (OrdinalEncoder)\n")
                        f.write("2. Trained DecisionTreeClassifier\n")
                        f.write("3. Extracted paths from root to leaves\n")
                        f.write("4. Each path = one rule\n\n")

                        f.write(f"RESULT: Generated {len(generated_rules)} rules\n")

                    elif self.selected_rule_method == "Forest":

                        f.write("-" * 50 + "\n")
                        f.write("RANDOM FOREST PARAMETERS (VARIABLE DEPTH)\n")
                        f.write("-" * 50 + "\n")
                        f.write(f"n_estimators (number of trees): {int(self.rf_n_estimators)}\n")
                        f.write(f"min_depth: {int(self.rf_min_depth)}\n")
                        f.write(f"max_depth: {int(self.rf_max_depth)}\n")
                        f.write(f"min_samples_leaf: {int(self.rf_min_samples_leaf)}\n")
                        f.write(f"random_state (base): {seed}\n")
                        f.write(f"max_features: 'sqrt' (each tree uses √n features)\n\n")

                        f.write("-" * 50 + "\n")
                        f.write("GENERATION PROCESS (FOR EACH TREE)\n")
                        f.write("-" * 50 + "\n")
                        f.write("For each of {} trees:\n".format(int(self.rf_n_estimators)))
                        f.write("  1. Random depth sampled from range [{}, {}]\n".format(
                            int(self.rf_min_depth), int(self.rf_max_depth)))
                        f.write("  2. Random subset of features (sqrt of all)\n")
                        f.write("  3. Training DecisionTreeClassifier\n")
                        f.write("  4. Extracting leaf paths as rules\n\n")

                        f.write("-" * 50 + "\n")
                        f.write("INDIVIDUAL TREE DETAILS\n")
                        f.write("-" * 50 + "\n")


                        if hasattr(runner, 'rule_generator') and hasattr(runner.rule_generator, 'estimators_'):
                            for i, tree in enumerate(runner.rule_generator.estimators_, 1):
                                actual_depth = tree.get_depth()
                                n_leaves = tree.get_n_leaves()
                                f.write(f"Tree {i}: actual_depth={actual_depth}, n_leaves={n_leaves}, random_state={seed + i}\n")
                            f.write("\n")
                        else:
                            f.write("(Detailed tree information unavailable)\n\n")

                        f.write(f"RESULT: Generated {len(generated_rules)} rules total\n")
                        avg_rules_per_tree = len(generated_rules) / int(self.rf_n_estimators) if int(self.rf_n_estimators) > 0 else 0
                        f.write(f"Average {avg_rules_per_tree:.1f} rules per tree\n")

                    elif self.selected_rule_method == "Naive":
                        f.write("-" * 50 + "\n")
                        f.write("NAIVE METHOD (Row-by-Row)\n")
                        f.write("-" * 50 + "\n")
                        f.write("Each dataset row = one rule\n")
                        f.write("All attributes = premises\n")
                        f.write("Decision column = conclusion\n\n")
                        f.write(f"RESULT: Generated {len(generated_rules)} rules\n")


                    f.write("\n" + "-" * 50 + "\n")
                    f.write("GENERATED RULES STATISTICS\n")
                    f.write("-" * 50 + "\n")

                    if generated_rules:
                        premise_counts = [len(r.premises) for r in generated_rules]
                        conclusions = [r.conclusion.value for r in generated_rules]
                        unique_conclusions = set(conclusions)

                        f.write(f"Number of rules: {len(generated_rules)}\n")
                        f.write(f"Min premises: {min(premise_counts)}\n")
                        f.write(f"Max premises: {max(premise_counts)}\n")
                        f.write(f"Average premises: {sum(premise_counts)/len(premise_counts):.2f}\n")
                        f.write(f"Unique conclusions: {len(unique_conclusions)}\n")
                        f.write(f"Conclusion values: {sorted(unique_conclusions)}\n")


                        from collections import Counter
                        conclusion_dist = Counter(conclusions)
                        f.write("\nConclusion distribution:\n")
                        for conc, count in conclusion_dist.most_common():
                            f.write(f"  {conc}: {count} rules ({100*count/len(generated_rules):.1f}%)\n")

                print(f"[BENCHMARK] Saved rule_generation_log.txt")
                print(f"[BENCHMARK] Results saved to: {benchmark_folder}")






                accuracy_by_strategy = {}
                for strategy_name in strategies:
                    strategy_results = [r for r in results if r["Strategy"] == strategy_name]
                    correct = sum(1 for r in strategy_results if r["Correctness"])
                    total = len(strategy_results)
                    accuracy_by_strategy[strategy_name] = round(correct / total * 100, 1) if total > 0 else 0


                accuracy_by_percent = {}
                for percent in knowledge_percentages:
                    percent_results = [r for r in results if r["Knowledge_Percent"] == percent]
                    correct = sum(1 for r in percent_results if r["Correctness"])
                    total = len(percent_results)
                    accuracy_by_percent[percent] = round(correct / total * 100, 1) if total > 0 else 0


                summary_rows = []


                for s, acc in accuracy_by_strategy.items():
                    summary_rows.append(
                        ft.Row([
                            ft.Text(f"{s}:", size=13, color=AppColors.TEXT_SECONDARY, width=120),
                            ft.Container(
                                content=ft.Text(f"{acc}%", size=13, color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                                bgcolor=ft.colors.with_opacity(0.2, AppColors.SECONDARY if acc >= 50 else AppColors.WARNING),
                                padding=ft.padding.symmetric(horizontal=8, vertical=2),
                                border_radius=4,
                            ),
                        ])
                    )


                summary_rows.append(ft.Divider(color=AppColors.BORDER, height=1))
                summary_rows.append(ft.Text("Accuracy vs % wiedzy:", size=12, color=AppColors.TEXT_MUTED, italic=True))
                for p, acc in accuracy_by_percent.items():
                    summary_rows.append(
                        ft.Row([
                            ft.Text(f"{p}% wiedzy:", size=13, color=AppColors.TEXT_SECONDARY, width=120),
                            ft.Container(
                                content=ft.Text(f"{acc}%", size=13, color=AppColors.TEXT_PRIMARY),
                                bgcolor=ft.colors.with_opacity(0.15, AppColors.PRIMARY),
                                padding=ft.padding.symmetric(horizontal=8, vertical=2),
                                border_radius=4,
                            ),
                        ])
                    )

                results_content = ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.icons.CHECK_CIRCLE_ROUNDED, color=AppColors.SECONDARY, size=24),
                            ft.Text("Benchmark zakończony!", size=16, color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                        ], spacing=12),

                        ft.Container(height=12),

                        ft.Text(f"Wykonano {total_runs} uruchomień", size=13, color=AppColors.TEXT_SECONDARY),
                        ft.Text(f"Wyniki zapisane do: results/benchmark_{timestamp}/", size=12, color=AppColors.TEXT_MUTED),

                        ft.Container(height=16),

                        ft.Text("Accuracy per strategia:", size=14, color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ft.Container(height=8),
                        *summary_rows,

                        ft.Container(height=16),

                        ft.ElevatedButton(
                            text="Otwórz folder wyników",
                            icon=ft.icons.FOLDER_OPEN_ROUNDED,
                            bgcolor=AppColors.BG_ELEVATED,
                            color=AppColors.TEXT_PRIMARY,
                            on_click=lambda e: os.startfile(benchmark_folder) if os.name == 'nt' else None,
                        ),
                    ], spacing=8),
                    padding=20,
                    border_radius=10,
                    bgcolor=AppColors.BG_ELEVATED,
                    border=ft.border.all(1, AppColors.SECONDARY),
                )

                self.inference_results_container.content = results_content


                self.next_button.disabled = False
                self.next_button.text = "Uruchom eksperyment"
                self.next_button.icon = ft.icons.PLAY_ARROW_ROUNDED

                if self.page:
                    self.page.update()

                self._show_snackbar(f"Benchmark zakończony! {total_runs} uruchomień. Wyniki w results/benchmark_{timestamp}/", success=True)




                def navigate_to_results_after_delay():
                    import time
                    time.sleep(1.5)


                    async def do_navigate():
                        if self.page and hasattr(self.page, 'sidebar'):
                            self.page.sidebar._on_click(2)

                    if self.page:
                        self.page.run_task(do_navigate)


                threading.Thread(target=navigate_to_results_after_delay, daemon=True).start()

            except Exception as e:
                print(f"[BENCHMARK ERROR] {e}")
                import traceback
                traceback.print_exc()

                self.benchmark_status_text.value = f"Błąd: {str(e)}"
                self.next_button.disabled = False
                self.next_button.text = "Spróbuj ponownie"
                self.next_button.icon = ft.icons.REFRESH_ROUNDED

                if self.page:
                    self.page.update()

                self._show_snackbar(f"Błąd benchmarku: {str(e)}", success=False)


        thread = threading.Thread(target=run_benchmark_thread, daemon=True)
        thread.start()

    def _create_facts_from_row(self, row) -> set:









        facts = set()

        for column, value in row.items():

            if pd.isna(value):
                continue


            value_str = str(value)


            fact = Fact(attribute=column, value=value_str)
            facts.add(fact)

        return facts

    def _display_inference_results(self, result, execution_time: float):








        results_content = ft.Column([

            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_ROUNDED, color=AppColors.SECONDARY if result.success else ft.colors.ORANGE, size=28),
                    ft.Text(
                        "Wyniki wnioskowania",
                        size=16,
                        color=AppColors.TEXT_PRIMARY,
                        weight=ft.FontWeight.W_600
                    ),
                ], spacing=12),
                margin=ft.margin.only(bottom=16)
            ),


            ft.Container(
                content=ft.Column([
                    self._create_metric_row("Status", "Sukces" if result.success else "Niepowodzenie",
                                          AppColors.SECONDARY if result.success else ft.colors.ORANGE),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_metric_row("Czas wykonania", f"{execution_time:.2f} ms", AppColors.PRIMARY),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_metric_row("Liczba kroków", str(result.iterations), AppColors.TEXT_PRIMARY),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_metric_row("Nowe fakty", str(len(result.new_facts)), AppColors.SECONDARY),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_metric_row("Łączna liczba faktów", str(result.facts_count), AppColors.TEXT_PRIMARY),
                    ft.Divider(color=AppColors.BORDER, height=1),
                    self._create_metric_row("Odpalone reguły", str(len(result.rules_fired)), AppColors.TEXT_PRIMARY),
                ], spacing=8),
                padding=20,
                border_radius=10,
                bgcolor=AppColors.BG_ELEVATED,
            ),

            ft.Container(height=16),


            ft.Container(
                content=ft.Column([
                    ft.Text("Przykładowe nowe fakty:", size=14, weight=ft.FontWeight.W_600, color=AppColors.TEXT_PRIMARY),
                    ft.Container(height=8),
                    *[
                        ft.Text(f"• {fact.attribute} = {fact.value}", size=12, color=AppColors.TEXT_SECONDARY)
                        for fact in list(result.new_facts)[:10]
                    ],
                    ft.Text(
                        f"... i {len(result.new_facts) - 10} więcej" if len(result.new_facts) > 10 else "",
                        size=11,
                        color=AppColors.TEXT_MUTED,
                        italic=True
                    ) if len(result.new_facts) > 10 else ft.Container(),
                ], spacing=4),
                padding=16,
                border_radius=8,
                bgcolor=ft.colors.with_opacity(0.05, AppColors.SECONDARY),
            ) if result.new_facts else ft.Container(),
        ], spacing=0)


        self.inference_results_container.content = results_content
        self._update_content()
        self.update()

    def _create_metric_row(self, label: str, value: str, value_color):

        return ft.Row([
            ft.Text(label, size=13, color=AppColors.TEXT_MUTED, width=180),
            ft.Text(value, size=13, color=value_color, weight=ft.FontWeight.W_600),
        ])

    def _on_file_picked(self, e: ft.FilePickerResultEvent):




        if e.files and len(e.files) > 0:
            file_path = e.files[0].path


            is_new_file = (self.loaded_file_path != file_path)


            self.selected_dataset = None


            self.loaded_file_path = file_path


            self._add_or_update_local_file(file_path)


            if is_new_file and self.current_step > 0:
                print(f"[RESET] Wykryto zmianę pliku, resetowanie stanu...")



                self._reset_experiment_state()


                self._update_stepper()
                self._update_content()
                self._update_navigation_buttons()

                print("[RESET] Stan wyczyszczony przez _reset_experiment_state()")


            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='windows-1250') as f:
                        self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
                except Exception:
                    self.csv_preview_lines = ["Nie można odczytać podglądu pliku"]
            except Exception as e:
                self.csv_preview_lines = [f"Błąd odczytu: {str(e)}"]


            self.current_step = 1
            self._update_stepper()
            self._update_content()
            self._update_navigation_buttons()
            self.update()

    def _validate_csv_with_config(self):

        if not self.loaded_file_path:
            return

        try:

            df, metadata = load_csv(
                self.loaded_file_path,
                column_separator=self.csv_column_separator,
                decimal_separator=self.csv_decimal_separator,
                has_header=self.csv_has_header,
                decision_column_index=-1,
                drop_missing=True,
                encoding=self.csv_encoding
            )


            self.loaded_df = df
            self.loaded_metadata = metadata


            filename = metadata['filename']
            rows = metadata['rows_final']
            cols = metadata['columns_total']


            status_rows = [
                ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text(
                        f"Plik '{filename}' został poprawnie wczytany ({rows} wierszy, {cols} kolumn)",
                        size=13,
                        color=AppColors.SECONDARY,
                        weight=ft.FontWeight.W_500
                    )
                ], spacing=8)
            ]


            removed_ids = metadata.get('removed_id_columns', [])
            if removed_ids:
                status_rows.append(
                    ft.Row([
                        ft.Icon(ft.icons.INFO_ROUNDED, color=AppColors.WARNING, size=18),
                        ft.Text(
                            f"Automatycznie usunięto kolumny indeksowe: {', '.join(removed_ids)}",
                            size=12,
                            color=AppColors.WARNING,
                        )
                    ], spacing=8)
                )

            self.file_status_container.content = ft.Column(status_rows, spacing=4)
            self.file_status_container.bgcolor = AppColors.SECONDARY_BG
            self.file_status_container.visible = True


            self._update_content()
            self.update()



            import threading
            import time

            expected_step = self.current_step

            def go_to_next_step():
                time.sleep(1)



                if self.current_step != expected_step:
                    print(f"[AUTO-ADVANCE] Anulowano - użytkownik już przeszedł (step: {self.current_step}, expected: {expected_step})")
                    return


                try:
                    self._go_next()
                except Exception as e:
                    print(f"[ERROR] Błąd podczas automatycznego przejścia: {e}")

            thread = threading.Thread(target=go_to_next_step, daemon=True)
            thread.start()

        except CSVLoadError as e:

            self.loaded_df = None
            self.loaded_metadata = None

            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text(
                    f"Błąd: {str(e)}",
                    size=13,
                    color=AppColors.ERROR,
                    weight=ft.FontWeight.W_500
                )
            ], spacing=8)
            self.file_status_container.bgcolor = AppColors.ERROR_BG
            self.file_status_container.visible = True


            self._update_content()
            self.update()

    def _select_dataset(self, dataset: str):







        self.loaded_file_path = None


        self.selected_dataset = dataset


        dataset_paths = {
            'Wine': 'Wine.csv',
            'Mushroom': 'Mushroom.csv',
            'Iris': 'Iris.csv',
            'Breast Cancer': 'BreastCancerWisconsin.csv',
            'Zoo': 'Zoo.csv',
            'Income': 'AdultIncome.csv',
            'Car': 'CarEvaluation.csv',
            'Indians Diabetes': 'PimaIndiansDiabetes.csv',
            'Ecoli': 'Ecoli.csv'
        }

        if dataset not in dataset_paths:
            print(f"[ERROR] Nieznany dataset: {dataset}")
            return



        file_path = resource_path(os.path.join('data', dataset_paths[dataset]))


        if not os.path.exists(file_path):
            print(f"[ERROR] Plik datasetu nie istnieje: {file_path}")

            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text(f"Nie znaleziono pliku: {dataset_paths[dataset]}", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self._update_content()
            self.update()
            return

        print(f"[DATASET] Wczytuję przykładowy dataset: {dataset} ({file_path})")


        self.loaded_file_path = file_path


        self._add_or_update_local_file(file_path)


        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='windows-1250') as f:
                    self.csv_preview_lines = [f.readline().strip() for _ in range(5)]
            except Exception:
                self.csv_preview_lines = ["Nie można odczytać podglądu pliku"]
        except Exception as e:
            self.csv_preview_lines = [f"Błąd odczytu: {str(e)}"]


        self.file_status_container.content = ft.Row([
            ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
            ft.Text(f"Wczytano dataset: {dataset}", size=13, color=AppColors.SECONDARY),
        ], spacing=8)
        self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
        self.file_status_container.visible = True


        self.current_step = 1
        self._update_stepper()
        self._update_content()
        self._update_navigation_buttons()
        self.update()

        print(f"[DATASET] Dataset wczytany pomyślnie: {dataset}")

    def _select_discretization(self, method: str):
        self.selected_discretization = method
        self._update_content()
        self.update()

    def _select_algorithm(self, algorithm: str):
        self.selected_algorithm = algorithm
        self._update_content()
        self.update()

    def _select_strategy(self, strategy: str):

        self.selected_strategy = strategy
        self._update_content()
        self.update()

    def _toggle_strategy(self, strategy: str):

        if strategy in self.selected_strategies:
            self.selected_strategies.discard(strategy)
        else:
            self.selected_strategies.add(strategy)


        if self.selected_strategies:
            self.selected_strategy = sorted(self.selected_strategies)[0]
        else:
            self.selected_strategy = None

        self._update_content()
        self.update()

    def _select_imputation(self, method_id: str):

        self.selected_imputation = method_id
        self._update_content()
        self.update()

    def _select_bins_choice(self, choice: str):






        self.bins_choice = choice


        if choice == "auto":
            recommended = self._calculate_recommended_bins()
            self.n_bins = recommended

            if self.bin_suggestion:
                method = self.bin_suggestion.recommended
                print(f"[AUTO BINS] Ustawiono liczbę binów na: {recommended} (metoda: {method})")
            else:
                print(f"[AUTO BINS] Ustawiono liczbę binów na: {recommended} (fallback)")


        self.steps = self._get_steps()
        self._update_stepper()
        self._update_content()
        self.update()

    def _toggle_column(self, column_name: str):

        if column_name in self.selected_columns:
            self.selected_columns.remove(column_name)
        else:
            self.selected_columns.add(column_name)
        self._update_content()
        self.update()

    def _select_all_columns(self):


        self.selected_columns = set([
            col for col in self.available_columns
            if col != self.csv_decision_column
        ])
        self._update_content()
        self.update()

    def _deselect_all_columns(self):

        self.selected_columns = set()
        self._update_content()
        self.update()

    def _calculate_recommended_bins(self) -> int:











        if self.loaded_df is None or len(self.loaded_df) == 0:
            self.bin_suggestion = None
            return 5


        numeric_cols = self.loaded_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            self.bin_suggestion = None
            return 5


        analysis_col = None
        for col in numeric_cols:
            if col != self.csv_decision_column:
                analysis_col = col
                break

        if analysis_col is None:

            analysis_col = numeric_cols[0]


        try:
            suggester = BinSuggester(min_bins=3, max_bins=20)
            suggestion = suggester.suggest(self.loaded_df[analysis_col])


            self.bin_suggestion = suggestion

            print(f"[SMART BINNING] Analiza kolumny: {analysis_col}")
            print(f"  - Sturges: {suggestion.sturges} | Scott: {suggestion.scott} | FD: {suggestion.freedman_diaconis}")
            print(f"  - Rekomendacja: {suggestion.recommended_bins} binów (metoda: {suggestion.recommended})")
            print(f"  - Powody: {', '.join(suggestion.reasons)}")

            return suggestion.recommended_bins

        except Exception as e:
            print(f"[ERROR] BinSuggester failed: {e}")
            self.bin_suggestion = None
            return 5

    def _execute_imputation(self) -> bool:










        print(f"[IMPUTATION] Rozpoczynam imputację...")
        print(f"[IMPUTATION] Wybrana metoda: {self.selected_imputation}")


        if self.loaded_df is None:
            print("[ERROR] Brak wczytanego pliku CSV")
            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text("Błąd: Brak wczytanego pliku CSV", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False

        try:
            rows_before = len(self.loaded_df)
            missing_before = self.loaded_df.isna().sum().sum()

            print(f"[IMPUTATION] Wiersze przed: {rows_before}, Brakujące wartości: {missing_before}")

            if self.selected_imputation == "remove_rows":

                print("[IMPUTATION] Usuwam wiersze z NaN...")
                self.loaded_df = self.loaded_df.dropna()

                rows_after = len(self.loaded_df)
                rows_removed = rows_before - rows_after

                print(f"[IMPUTATION] Usunięto {rows_removed} wierszy")


                self.file_status_container.content = ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text(f"Usunięto {rows_removed} wierszy z brakującymi wartościami",
                           size=13, color=AppColors.SECONDARY),
                ], spacing=8)
                self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
                self.file_status_container.visible = True

            elif self.selected_imputation == "smart_fill":

                print(f"[IMPUTATION] Smart Fill - numeric: {self.imputation_numeric_method}, categorical: {self.imputation_categorical_method}")


                from preprocessing.imputer import Imputer


                if self.csv_decision_column and self.csv_decision_column in self.loaded_df.columns:
                    decision_column_name = self.csv_decision_column
                else:

                    decision_column_name = self.loaded_df.columns[-1]

                print(f"[IMPUTATION] Kolumna decyzyjna: {decision_column_name}")


                imputer = Imputer()


                self.loaded_df, report = imputer.impute(
                    self.loaded_df,
                    decision_column=decision_column_name,
                    numeric_method=self.imputation_numeric_method,
                    categorical_method=self.imputation_categorical_method,
                    columns=None
                )


                self.imputation_report = report


                print(f"[IMPUTATION] Raport:")
                print(f"  - Brakujących przed: {report.total_missing}")
                print(f"  - Kolumn z imputacją: {len(report.columns_affected)}")
                print(f"  - Wartości uzupełnione: {sum(report.values_imputed.values())}")
                print(f"  - Metoda: {report.method_used}")
                if report.columns_affected:
                    print(f"  - Kolumny: {', '.join(report.columns_affected)}")


                total_imputed = sum(report.values_imputed.values())
                cols_count = len(report.columns_affected)
                imputed_summary = f"Uzupełniono {total_imputed} wartości w {cols_count} kolumnach" if cols_count > 0 else "Brak wartości do uzupełnienia"

                self.file_status_container.content = ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text(imputed_summary, size=13, color=AppColors.SECONDARY),
                ], spacing=8)
                self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
                self.file_status_container.visible = True

            else:
                print(f"[ERROR] Nieznana metoda imputacji: {self.selected_imputation}")
                return False

            missing_after = self.loaded_df.isna().sum().sum()
            print(f"[IMPUTATION] Zakończono! Brakujące wartości po: {missing_after}")


            self.imputation_completed = True

            return True

        except Exception as e:
            print(f"[ERROR] Błąd imputacji: {e}")
            import traceback
            traceback.print_exc()


            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text(f"Błąd imputacji: {str(e)}", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False

    def _execute_discretization(self) -> bool:






        print("\n" + "="*60)
        print("KROK 4: DYSKRETYZACJA")
        print("="*60)


        if not getattr(self, 'imputation_completed', False):
            print("[ERROR] Imputacja nie została ukończona - zatrzymuję dyskretyzację")
            self._show_snackbar("Błąd: Najpierw wykonaj imputację", success=False)
            return False

        print(f"[DISCRETIZATION] Rozpoczynam dyskretyzację...")
        print(f"[DISCRETIZATION] Metoda: {self.selected_discretization}")
        print(f"[DISCRETIZATION] Liczba binów: {self.n_bins}")
        print(f"[DISCRETIZATION] Bins choice: {self.bins_choice}")



        if self.selected_discretization != "Brak" and self.bins_choice == "manual" and len(self.selected_columns) == 0:
            print("[ERROR] Brak zaznaczonych kolumn do dyskretyzacji (tryb manual)")

            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text("Błąd: Zaznacz przynajmniej jedną kolumnę do dyskretyzacji",
                       size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False


        if self.loaded_df is None:
            print("[ERROR] Brak wczytanego pliku CSV")

            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text("Błąd: Brak wczytanego pliku CSV", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False

        try:

            from discretization import (
                EqualWidthDiscretizer,
                EqualFrequencyDiscretizer,
                DiscretizationError
            )


            if self.selected_discretization == "Brak":
                print("[DISCRETIZATION] Metoda 'Brak' - kopiuję DataFrame bez zmian")
                self.discretized_df = self.loaded_df.copy()
                print(f"[DISCRETIZATION] Zakończono (brak dyskretyzacji)")
                self.discretization_completed = True
                return True



            self.discretized_df = self.loaded_df.copy()


            if not self.selected_columns or len(self.selected_columns) == 0:

                auto_numeric_cols = [
                    col for col in self.loaded_df.columns
                    if pd.api.types.is_numeric_dtype(self.loaded_df[col])
                ]
                self.selected_columns = set(auto_numeric_cols)
                print(f"[DISCRETIZATION] Auto-select: Wybrano domyślnie wszystkie kolumny numeryczne ({len(auto_numeric_cols)} kolumn)")



            numeric_columns = []
            for col in self.loaded_df.columns:
                if col in self.selected_columns:

                    if pd.api.types.is_numeric_dtype(self.loaded_df[col]):
                        numeric_columns.append(col)

            if len(numeric_columns) == 0:
                print("[WARNING] Brak numerycznych kolumn do dyskretyzacji")

                print(f"[DISCRETIZATION] Zakończono (brak numerycznych kolumn)")
                self.discretization_completed = True
                return True

            print(f"[DISCRETIZATION] Kolumny do dyskretyzacji: {numeric_columns}")


            if self.selected_discretization == "Equal Width":
                discretizer = EqualWidthDiscretizer(n_bins=int(self.n_bins))
            elif self.selected_discretization == "Equal Frequency":
                discretizer = EqualFrequencyDiscretizer(n_bins=int(self.n_bins))
            elif self.selected_discretization == "K-Means":

                print("[WARNING] K-Means nie jest jeszcze zaimplementowany - używam Equal Width")

                self._show_snackbar(
                    "Metoda K-Means nie jest jeszcze w pełni zaimplementowana. Użyto Equal Width.",
                    success=False
                )
                discretizer = EqualWidthDiscretizer(n_bins=int(self.n_bins))
            else:
                print(f"[ERROR] Nieznana metoda dyskretyzacji: {self.selected_discretization}")
                return False


            for col in numeric_columns:
                print(f"[DISCRETIZATION] Dyskretyzuję kolumnę: {col}")


                data = self.discretized_df[col].values


                discretized_values = discretizer.fit_transform(data)



                bin_labels = [f"Bin_{i}" for i in range(int(self.n_bins))]


                discretized_labels = []
                for val in discretized_values:
                    if val == -1:
                        discretized_labels.append("Unknown")
                    else:
                        discretized_labels.append(bin_labels[val])


                self.discretized_df[col] = discretized_labels

                print(f"[DISCRETIZATION] Kolumna {col} zdyskretyzowana ({len(set(discretized_labels))} unikalnych wartości)")

            print(f"[DISCRETIZATION] Dyskretyzacja zakończona pomyślnie")


            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
                ft.Text(f"Dyskretyzacja zakończona ({self.selected_discretization}, {int(self.n_bins)} binów)",
                       size=13, color=AppColors.SECONDARY),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
            self.file_status_container.visible = True

            self.discretization_completed = True
            return True

        except DiscretizationError as e:
            print(f"[ERROR] Błąd dyskretyzacji: {e}")

            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text(f"Błąd dyskretyzacji: {str(e)}", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False

        except Exception as e:
            print(f"[ERROR] Nieoczekiwany błąd: {e}")
            import traceback
            traceback.print_exc()

            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text(f"Błąd: {str(e)}", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False

    def _execute_rule_generation(self) -> bool:










        print("\n" + "="*60)
        print("TICKET #2: GENEROWANIE REGUŁ")
        print("="*60)

        print(f"[RULE GEN] Rozpoczynam generowanie reguł...")
        print(f"[RULE GEN] Metoda: {self.selected_rule_method}")


        if self.discretized_df is None:
            print("[ERROR] Brak zdyskretyzowanych danych - wymagany krok dyskretyzacji")
            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text("Błąd: Brak zdyskretyzowanych danych", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False


        if self.csv_decision_column and self.csv_decision_column in self.discretized_df.columns:
            decision_column = self.csv_decision_column
        else:

            decision_column = self.discretized_df.columns[-1]

        print(f"[RULE GEN] Kolumna decyzyjna: {decision_column}")

        try:
            if self.selected_rule_method == "Naive":
              
                from preprocessing.rule_generator import RuleGenerator

                print(f"[RULE GEN] Metoda Naive - każdy wiersz staje się regułą")
                rule_gen = RuleGenerator()
                self.generated_rules = rule_gen.generate(
                    self.discretized_df,
                    decision_column=decision_column
                )

                rules_count = len(self.generated_rules)
                print(f"[RULE GEN] Wygenerowano {rules_count} reguł")

                if rules_count > 0:
                    print(f"[RULE GEN] Przykładowa reguła:")
                    print(f"  {self.generated_rules[0]}")


                self.file_status_container.content = ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text(f"Wygenerowano {rules_count} reguł (Naive)",
                           size=13, color=AppColors.SECONDARY),
                ], spacing=8)
                self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
                self.file_status_container.visible = True

                return True

            elif self.selected_rule_method == "Tree":
              
                from preprocessing.tree_rule_generator import TreeRuleGenerator

                print(f"[RULE GEN] Tree Config:")
                print(f"  - max_depth: {self.tree_max_depth}")
                print(f"  - min_samples_leaf: {self.tree_min_samples_leaf}")

                rule_gen = TreeRuleGenerator(
                    max_depth=int(self.tree_max_depth),
                    min_samples_leaf=int(self.tree_min_samples_leaf)
                )

                self.generated_rules = rule_gen.generate(
                    self.discretized_df,
                    decision_column=decision_column
                )

                rules_count = len(self.generated_rules)
                print(f"[RULE GEN] Wygenerowano {rules_count} reguł")

                if rules_count > 0:
                    print(f"[RULE GEN] Przykładowa reguła:")
                    print(f"  {self.generated_rules[0]}")


                self.file_status_container.content = ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text(f"Wygenerowano {rules_count} reguł (Tree)",
                           size=13, color=AppColors.SECONDARY),
                ], spacing=8)
                self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
                self.file_status_container.visible = True

                return True

            elif self.selected_rule_method == "Forest":
              
                from preprocessing.forest_rule_generator import ForestRuleGenerator

                print(f"[RULE GEN] Forest Config (Variable Depth):")
                print(f"  - n_estimators: {self.rf_n_estimators}")
                print(f"  - min_depth: {self.rf_min_depth}")
                print(f"  - max_depth: {self.rf_max_depth}")
                print(f"  - min_samples_leaf: {self.rf_min_samples_leaf}")

                rule_gen = ForestRuleGenerator(
                    n_estimators=int(self.rf_n_estimators),
                    min_depth=int(self.rf_min_depth),
                    max_depth=int(self.rf_max_depth),
                    min_samples_leaf=int(self.rf_min_samples_leaf)
                )

                self.generated_rules = rule_gen.generate(
                    self.discretized_df,
                    decision_column=decision_column
                )

                rules_count = len(self.generated_rules)
                print(f"[RULE GEN] Wygenerowano {rules_count} reguł")

                if rules_count > 0:
                    print(f"[RULE GEN] Przykładowa reguła:")
                    print(f"  {self.generated_rules[0]}")


                self.file_status_container.content = ft.Row([
                    ft.Icon(ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=AppColors.SECONDARY, size=20),
                    ft.Text(f"Wygenerowano {rules_count} reguł (Forest)",
                           size=13, color=AppColors.SECONDARY),
                ], spacing=8)
                self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.SECONDARY)
                self.file_status_container.visible = True

                return True

            else:
                print(f"[ERROR] Nieznana metoda generowania reguł: {self.selected_rule_method}")
                return False

        except Exception as e:
            print(f"[ERROR] Błąd generowania reguł: {e}")
            import traceback
            traceback.print_exc()


            self.file_status_container.content = ft.Row([
                ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                ft.Text(f"Błąd generowania reguł: {str(e)}", size=13, color=AppColors.ERROR),
            ], spacing=8)
            self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
            self.file_status_container.visible = True
            self.update()
            return False

    def _next_step(self, e):
        current_step_name = self.steps[self.current_step]



        if current_step_name == lang.t('new_exp_step_run'):

            self.next_button.disabled = True
            self.next_button.text = "Pracuję..."
            self.next_button.icon = ft.icons.HOURGLASS_EMPTY_ROUNDED
            self.update()


            def run_inference_thread():
                try:

                    self._execute_inference()


                    self.next_button.disabled = True
                    self.next_button.text = "Gotowe!"
                    self.next_button.icon = ft.icons.CHECK_CIRCLE_ROUNDED
                    self.next_button.bgcolor = AppColors.SECONDARY

                    if self.page:
                        self.page.update()
                except Exception as e:
                    print(f"[ERROR] Błąd wątku wnioskowania: {e}")



            thread = threading.Thread(target=run_inference_thread, daemon=True)
            thread.start()
            return


        if self.current_step < len(self.steps) - 1:

            if current_step_name == "Konfiguracja eksperymentu":
                is_valid, error_msg = self._validate_seed(self.random_seed)
                if not is_valid:
                    print(f"[ERROR] Niepoprawny Random Seed: {error_msg}")
                    self._show_snackbar(lang.t('error_invalid_seed') if hasattr(lang, 't') and lang.t('error_invalid_seed') != 'error_invalid_seed' else "Niepoprawny Random Seed. Wymagana liczba całkowita nieujemna (np. 42).", success=False)
                    return


            if current_step_name == lang.t('new_exp_step_imputation'):

                if self.selected_imputation is None:
                    print("[ERROR] Brak wyboru metody imputacji")
                    self._show_snackbar(lang.t('error_select_imputation'), success=False)
                    return


                success = self._execute_imputation()
                if not success:

                    self._show_snackbar("Błąd podczas imputacji. Sprawdź dane.", success=False)
                    return


            if current_step_name == lang.t('new_exp_step_discretization'):

                if not getattr(self, 'imputation_completed', False):
                    print("[ERROR] Próba przejścia do dyskretyzacji bez ukończonej imputacji")
                    self._show_snackbar("Najpierw ukończ krok imputacji", success=False)
                    return

                if self.selected_discretization is None:
                    print("[ERROR] Brak wyboru metody dyskretyzacji")
                    self._show_snackbar("Wybierz metodę dyskretyzacji przed przejściem dalej", success=False)
                    return


                if self.selected_discretization != "Brak" and self.bins_choice is None:
                    print("[ERROR] Brak wyboru auto/manual dla liczby binów")
                    self._show_snackbar("Wybierz sposób ustawienia liczby binów (auto/manual)", success=False)
                    return


                success = self._execute_discretization()
                if not success:

                    return

          
            if current_step_name == lang.t('new_exp_step_rule_generation'):

                if self.selected_rule_method is None:
                    print("[ERROR] Brak wyboru metody generowania reguł")
                    self._show_snackbar("Wybierz metodę generowania reguł przed przejściem dalej", success=False)
                    return


                if self.selected_rule_method == "Tree":
                    if hasattr(self, 'tree_max_depth_error') and self.tree_max_depth_error:
                        self._show_snackbar(f"Błąd parametru Tree Max Depth: {self.tree_max_depth_error}", success=False)
                        return
                    if hasattr(self, 'tree_min_samples_leaf_error') and self.tree_min_samples_leaf_error:
                        self._show_snackbar(f"Błąd parametru Tree Min Samples Leaf: {self.tree_min_samples_leaf_error}", success=False)
                        return


                if self.selected_rule_method == "Forest":
                    if hasattr(self, 'rf_min_depth_error') and self.rf_min_depth_error:
                        self._show_snackbar(f"Błąd parametru Forest Min Depth: {self.rf_min_depth_error}", success=False)
                        return
                    if hasattr(self, 'rf_max_depth_error') and self.rf_max_depth_error:
                        self._show_snackbar(f"Błąd parametru Forest Max Depth: {self.rf_max_depth_error}", success=False)
                        return
                    if hasattr(self, 'rf_min_samples_leaf_error') and self.rf_min_samples_leaf_error:
                        self._show_snackbar(f"Błąd parametru Forest Min Samples Leaf: {self.rf_min_samples_leaf_error}", success=False)
                        return
                    if hasattr(self, 'rf_n_estimators_error') and self.rf_n_estimators_error:
                        self._show_snackbar(f"Błąd parametru Forest N Estimators: {self.rf_n_estimators_error}", success=False)
                        return



                self.next_button.disabled = True
                self.next_button.text = "Generowanie reguł..."
                self.next_button.icon = ft.icons.HOURGLASS_EMPTY_ROUNDED
                self.update()

                def run_rule_generation_thread():
                    try:

                        success = self._execute_rule_generation()
                        print(f"[RULE GEN THREAD] Generowanie zakończone, success={success}")


                        import asyncio

                        async def update_ui_on_main_thread():
                            print("[RULE GEN ASYNC] Rozpoczynam aktualizację UI na głównym wątku...")
                            if success:

                                self.current_step += 1
                                print(f"[RULE GEN ASYNC] current_step = {self.current_step}")

                                self._update_stepper()
                                self._update_content()
                                self._update_navigation_buttons()


                                self.next_button.disabled = False
                                self.next_button.text = lang.t('next')
                                self.next_button.icon = ft.icons.ARROW_FORWARD_ROUNDED
                                print("[RULE GEN ASYNC] UI zaktualizowane pomyślnie")
                            else:

                                self.next_button.disabled = False
                                self.next_button.text = lang.t('next')
                                self.next_button.icon = ft.icons.ARROW_FORWARD_ROUNDED
                                print("[RULE GEN ASYNC] Błąd - przycisk przywrócony")

                            self.update()


                        if self.page:
                            self.page.run_task(update_ui_on_main_thread)
                        else:
                            print("[RULE GEN THREAD] UWAGA: self.page is None!")

                    except Exception as e:
                        print(f"[ERROR] Błąd wątku generowania reguł: {e}")
                        import traceback
                        traceback.print_exc()

                        async def restore_button_async():
                            self.next_button.disabled = False
                            self.next_button.text = lang.t('next')
                            self.next_button.icon = ft.icons.ARROW_FORWARD_ROUNDED
                            self.update()

                        if self.page:
                            self.page.run_task(restore_button_async)


                thread = threading.Thread(target=run_rule_generation_thread, daemon=True)
                thread.start()
                return

          
            if current_step_name == lang.t('new_exp_step_algorithm'):

                if self.selected_algorithm == "Backward Chaining":
                    if not self.backward_goal_attr:
                        print("[ERROR] Backward Chaining wymaga wyboru atrybutu celu")
                        self.file_status_container.content = ft.Row([
                            ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                            ft.Text("Błąd: Wybierz atrybut celu",
                                   size=13, color=AppColors.ERROR),
                        ], spacing=8)
                        self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
                        self.file_status_container.visible = True
                        self.update()
                        return

                    if not self.backward_goal_any_value and not self.backward_goal_value:
                        print("[ERROR] Backward Chaining wymaga wyboru wartości celu (lub zaznacz 'Dowolna wartość')")
                        self.file_status_container.content = ft.Row([
                            ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                            ft.Text("Błąd: Wybierz wartość celu lub zaznacz 'Dowolna wartość'",
                                   size=13, color=AppColors.ERROR),
                        ], spacing=8)
                        self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
                        self.file_status_container.visible = True
                        self.update()
                        return


                if self.selected_algorithm == "Forward Chaining" and self.use_forward_goal:
                    if not self.forward_goal_attr:
                        print("[ERROR] Forward z celem wymaga wyboru atrybutu celu")
                        self.file_status_container.content = ft.Row([
                            ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                            ft.Text("Błąd: Wybierz atrybut celu",
                                   size=13, color=AppColors.ERROR),
                        ], spacing=8)
                        self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
                        self.file_status_container.visible = True
                        self.update()
                        return

                    if not self.forward_goal_any_value and not self.forward_goal_value:
                        print("[ERROR] Forward z celem wymaga wyboru wartości celu (lub zaznacz 'Dowolna wartość')")
                        self.file_status_container.content = ft.Row([
                            ft.Icon(ft.icons.ERROR_OUTLINE_ROUNDED, color=AppColors.ERROR, size=20),
                            ft.Text("Błąd: Wybierz wartość celu lub zaznacz 'Dowolna wartość'",
                                   size=13, color=AppColors.ERROR),
                        ], spacing=8)
                        self.file_status_container.bgcolor = ft.colors.with_opacity(0.1, AppColors.ERROR)
                        self.file_status_container.visible = True
                        self.update()
                        return



                goal = None
                if self.selected_algorithm == "Backward Chaining":
                    if self.backward_goal_any_value:

                        goal = self.backward_goal_attr
                    else:

                        goal = (self.backward_goal_attr, self.backward_goal_value)
                elif self.selected_algorithm == "Forward Chaining" and self.use_forward_goal:
                    if self.forward_goal_any_value:

                        goal = self.forward_goal_attr
                    else:

                        goal = (self.forward_goal_attr, self.forward_goal_value)

                self.inference_config = {
                    "method": "forward" if self.selected_algorithm == "Forward Chaining" else "backward",
                    "use_clustering": self.use_clustering,
                    "use_greedy": self.use_greedy,
                    "goal": goal,
                }
                print(f"[ALGORITHM] Konfiguracja zapisana: {self.inference_config}")


            if current_step_name == lang.t('new_exp_step_strategy'):

                if not self.selected_strategies:
                    print("[ERROR] Brak wyboru strategii rozwiązywania konfliktów")
                    self._show_snackbar("Wybierz co najmniej jedną strategię rozwiązywania konfliktów", success=False)
                    return


            self.current_step += 1
            self._update_stepper()
            self._update_content()
            self._update_navigation_buttons()
            self.update()

    def _prev_step(self, e):

        if self.current_step > 0:

            self._clear_step_state(self.current_step)

            self.current_step -= 1
            self._update_stepper()
            self._update_content()
            self._update_navigation_buttons()
            self.update()
            print(f"[NAVIGATION] Cofnięto do kroku {self.current_step}, wyczyszczono stan kroku {self.current_step + 1}")

    def _restart_experiment(self, e):

        print("[RESTART] Rozpoczynam od nowa - reset całego stanu")


        self._reset_experiment_state()


        self.current_step = 0


        self._update_stepper()
        self._update_content()
        self._update_navigation_buttons()
        self.update()

        self._show_snackbar("Eksperyment zresetowany. Rozpocznij konfigurację od nowa.", success=True)

    def _clear_step_state(self, step_index: int):






        step_name = self.steps[step_index] if step_index < len(self.steps) else None
        print(f"[CLEAR] Czyszczenie stanu kroku {step_index}: {step_name}")


        if step_index == 0:

            pass


        elif step_name == lang.t('new_exp_step_data'):
            self.loaded_file_path = None
            self.loaded_df = None
            self.loaded_metadata = None
            self.selected_dataset = None
            self.csv_preview_lines = []


        elif step_name == lang.t('new_exp_step_csv_config'):
            self.csv_column_separator = ','
            self.csv_decimal_separator = '.'
            self.csv_has_header = True
            self.csv_decision_column = None


        elif step_name == lang.t('new_exp_step_imputation'):
            self.selected_imputation = None
            self.imputation_report = None
            self.imputation_completed = False


        elif step_name == lang.t('new_exp_step_discretization'):
            self.selected_discretization = None
            self.bins_choice = None
            self.discretization_completed = False


        elif step_name == lang.t('new_exp_step_disc_details'):
            self.n_bins = 5
            self.selected_columns = set()
            self.disc_details_initialized = False


        elif step_name == lang.t('new_exp_step_rule_generation'):
            self.selected_rule_method = None
            self.generated_rules = None

            self.tree_max_depth = 3
            self.tree_min_samples_leaf = 5

            self.rf_min_depth = 2
            self.rf_max_depth = 12
            self.rf_min_samples_leaf = 5
            self.rf_n_estimators = 100

            self.tree_max_depth_error = None
            self.tree_min_samples_leaf_error = None
            self.rf_min_depth_error = None
            self.rf_max_depth_error = None
            self.rf_min_samples_leaf_error = None
            self.rf_n_estimators_error = None


        elif step_name == lang.t('new_exp_step_algorithm'):
            self.selected_algorithm = "Forward Chaining"
            self.use_clustering = False
            self.n_clusters = 10
            self.centroid_method = "specialized"
            self.centroid_threshold = 0.3
            self.centroid_match_threshold = 0.0
            self.use_greedy = False
            self.use_forward_goal = False
            self.forward_goal_attr = None
            self.forward_goal_value = None
            self.forward_goal_any_value = False
            self.backward_goal_attr = None
            self.backward_goal_value = None
            self.backward_goal_any_value = False
            self.inference_config = {}


        elif step_name == lang.t('new_exp_step_strategy'):
            self.selected_strategy = None
            self.selected_strategies = set()


        elif step_name == lang.t('new_exp_step_run'):
            self.initial_facts_percent = "10, 25, 50"
            self.repetitions = "50"
            self.facts_validation_error = None
            self.repetitions_validation_error = None
            if hasattr(self, 'inference_results_container') and self.inference_results_container:
                self.inference_results_container.content = None

    def _go_next(self):




        self.current_step += 1
        self._update_stepper()
        self._update_content()
        self._update_navigation_buttons()
        self.update()
        print(f"[NAVIGATION] Przejście do kroku {self.current_step}")

    def _jump_to_step(self, step_index: int):





        if step_index > self.current_step:
            print(f"[NAVIGATION] Zablokowano skok do kroku {step_index} (obecny: {self.current_step})")
            self._show_snackbar("Ukończ bieżący krok aby przejść dalej", success=False)
            return


        self.current_step = step_index
        self._update_stepper()
        self._update_content()
        self._update_navigation_buttons()
        self.update()


class KnowledgeBaseView(ft.UserControl):
    def __init__(self, on_navigate=None):
        super().__init__()


        self.on_navigate = on_navigate


        self.firebase = FirebaseService()


        self.local_files_path = os.path.join(
            os.path.dirname(__file__),
            'local_files.json'
        )


        self.local_files = self._load_local_files()

        self.firebase_files = []

        self.assets_files = [
            {"name": "AdultIncome", "location": lang.t('kb_location_assets'), "size": "3.8 MB", "ext": ".csv", "attrs": "15", "rows": "32562", "date_added": "-"},
            {"name": "BreastCancerWisconsin", "location": lang.t('kb_location_assets'), "size": "16 KB", "ext": ".csv", "attrs": "10", "rows": "700", "date_added": "-"},
            {"name": "CarEvaluation", "location": lang.t('kb_location_assets'), "size": "51 KB", "ext": ".csv", "attrs": "7", "rows": "1729", "date_added": "-"},
            {"name": "CreditApproval", "location": lang.t('kb_location_assets'), "size": "32 KB", "ext": ".csv", "attrs": "16", "rows": "690", "date_added": "-"},
            {"name": "Ecoli", "location": lang.t('kb_location_assets'), "size": "16 KB", "ext": ".csv", "attrs": "9", "rows": "337", "date_added": "-"},
            {"name": "GlassIdentification", "location": lang.t('kb_location_assets'), "size": "12 KB", "ext": ".csv", "attrs": "11", "rows": "215", "date_added": "-"},
            {"name": "HeartDisease", "location": lang.t('kb_location_assets'), "size": "18 KB", "ext": ".csv", "attrs": "14", "rows": "303", "date_added": "-"},
            {"name": "Iris", "location": lang.t('kb_location_assets'), "size": "4.5 KB", "ext": ".csv", "attrs": "5", "rows": "151", "date_added": "-"},
            {"name": "Mushroom", "location": lang.t('kb_location_assets'), "size": "365 KB", "ext": ".csv", "attrs": "23", "rows": "8124", "date_added": "-"},
            {"name": "PimaIndiansDiabetes", "location": lang.t('kb_location_assets'), "size": "23 KB", "ext": ".csv", "attrs": "9", "rows": "768", "date_added": "-"},
            {"name": "Titanic", "location": lang.t('kb_location_assets'), "size": "59 KB", "ext": ".csv", "attrs": "12", "rows": "891", "date_added": "-"},
            {"name": "Wine", "location": lang.t('kb_location_assets'), "size": "11 KB", "ext": ".csv", "attrs": "14", "rows": "179", "date_added": "-"},
            {"name": "Zoo", "location": lang.t('kb_location_assets'), "size": "4.2 KB", "ext": ".csv", "attrs": "18", "rows": "102", "date_added": "-"},
        ]


        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)


        self.search_query = ""


        self.local_files_container = ft.Container()
        self.firebase_files_container = ft.Container()
        self.assets_files_container = ft.Container()

    def did_mount(self):


        if self.firebase.is_logged_in():
            print("[INFO] Automatyczne ładowanie plików z Firebase...")
            self._on_refresh_clicked("firebase")

    def build(self):

        self._refresh_containers()

        return ft.Column([
            self.file_picker,


            ft.Row([
                ft.Column([
                    ft.Text(lang.t('kb_title'), size=28, weight=ft.FontWeight.BOLD,
                           color=AppColors.TEXT_PRIMARY),
                    ft.Text(lang.t('kb_subtitle'), size=14,
                           color=AppColors.TEXT_SECONDARY),
                ], spacing=4),
                ft.Container(expand=True),
                ft.TextField(
                    hint_text=lang.t('kb_search_rules'),
                    prefix_icon=ft.icons.SEARCH_ROUNDED,
                    border_color=AppColors.BORDER,
                    focused_border_color=AppColors.PRIMARY,
                    width=300,
                    on_change=lambda e: self._on_search_change(e.control.value),
                ),
            ]),

            ft.Container(height=20),


            self.local_files_container,

            ft.Container(height=20),


            self.firebase_files_container,

            ft.Container(height=20),


            self.assets_files_container,
        ], scroll=ft.ScrollMode.AUTO)

    def _refresh_containers(self):


        self.local_files_container.content = create_card(
            self._create_data_table_section(
                title=lang.t('kb_data_sources'),
                file_list=self.local_files,
                list_name="local",
                show_checkboxes=True,
                show_delete_button=True,
            )
        )


        self.firebase_files_container.content = create_card(
            self._create_firebase_section()
        )


        self.assets_files_container.content = create_card(
            self._create_data_table_section(
                title=lang.t('kb_assets_files'),
                file_list=self.assets_files,
                list_name="assets",
                show_checkboxes=False,
                show_delete_button=False,
            )
        )
    
    def _on_search_change(self, value: str):

        self.search_query = value.lower()


    def _load_local_files(self) -> list:

        if os.path.exists(self.local_files_path):
            try:
                with open(self.local_files_path, 'r', encoding='utf-8') as f:
                    files = json.load(f)


                for file in files:
                    file['selected'] = False

                print(f"[OK] Wczytano {len(files)} lokalnych plików z pamięci")
                return files
            except Exception as e:
                print(f"[ERROR] Błąd wczytywania lokalnych plików: {e}")
                return []
        else:
            print("[INFO] Brak zapisanych lokalnych plików - tworzę pustą listę")
            return []

    def _save_local_files(self):

        try:

            files_to_save = []
            for file in self.local_files:
                file_copy = file.copy()
                file_copy.pop('selected', None)
                files_to_save.append(file_copy)

            with open(self.local_files_path, 'w', encoding='utf-8') as f:
                json.dump(files_to_save, f, indent=2, ensure_ascii=False)
            print(f"[OK] Zapisano {len(files_to_save)} lokalnych plików do pamięci")
        except Exception as e:
            print(f"[ERROR] Błąd zapisywania lokalnych plików: {e}")

    def _on_file_picked(self, e: ft.FilePickerResultEvent):

        if e.files and len(e.files) > 0:
            file = e.files[0]
            original_file_name = file.name
            file_path = file.path

            import os
            import datetime


            file_size = os.path.getsize(file_path)
            size_mb = f"{file_size / (1024 * 1024):.1f} MB"
            date_added = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


            local_names = [f["name"] for f in self.local_files]
            file_name = original_file_name

            if file_name in local_names:
                print(f"[WARNING] Plik '{file_name}' już istnieje w lokalnej pamięci")


                if not self.firebase.is_logged_in():
                    print(f"[INFO] Plik '{file_name}' został pominięty - już istnieje")
                    return


            if self.firebase.is_logged_in():

                if self.firebase.file_exists(original_file_name):
                    print(f"[ERROR] Plik '{original_file_name}' już istnieje w Firebase!")
                    print(f"[INFO] Nie można dodać duplikatu do chmury")


                    if file_name not in local_names:
                        ext = os.path.splitext(file_name)[1]
                        new_local_file = {
                            "name": file_name,
                            "path": file_path,
                            "size": size_mb,
                            "ext": ext,
                            "attrs": "?",
                            "rows": "?",
                            "date_added": date_added,
                            "selected": False,
                        }
                        self.local_files.append(new_local_file)
                        print(f"[INFO] Plik dodany tylko do lokalnej pamięci")


                    self._refresh_containers()
                    self.update()
                    return


            ext = os.path.splitext(file_name)[1]


            if file_name not in local_names:
                new_local_file = {
                    "name": file_name,
                    "path": file_path,
                    "size": size_mb,
                    "ext": ext,
                    "attrs": "?",
                    "rows": "?",
                    "date_added": date_added,
                    "selected": False,
                }
                self.local_files.append(new_local_file)


            if self.firebase.is_logged_in():
                success = self.firebase.upload_file(file_path, file_name)
                if success:

                    self._on_refresh_clicked("firebase")
                    print(f"[OK] Plik '{file_name}' dodany do Firebase i lokalnej pamięci")
                else:
                    print(f"[ERROR] Nie udało się przesłać pliku '{file_name}' do Firebase, dodano tylko lokalnie")
            else:
                print(f"[INFO] Plik '{file_name}' dodany tylko do lokalnej pamięci (użytkownik nie zalogowany)")


            self._save_local_files()


            self._refresh_containers()
            self.update()

    def _filter_files(self, files: list) -> list:

        if not self.search_query:
            return files
        return [f for f in files if self.search_query in f["name"].lower()]

    def _on_checkbox_change(self, file_list: list, file_index: int, value: bool):

        file_list[file_index]["selected"] = value

        self._refresh_containers()
        self.update()

    def _on_delete_clicked(self, file_list: list, list_name: str):


        selected_files = [f for f in file_list if f.get("selected", False)]

        if not selected_files:
            return


        self._show_delete_confirmation(file_list, selected_files, list_name)

    def _show_delete_confirmation(self, file_list: list, selected_files: list, list_name: str):

        def on_confirm(e):

            for file in selected_files:

                if list_name == "firebase" and "firebase_id" in file:
                    firebase_id = file.get("firebase_id")
                    if firebase_id:
                        success = self.firebase.delete_file(firebase_id)
                        if success:
                            print(f"[OK] Plik '{file['name']}' usunięty z Firebase")
                        else:
                            print(f"[ERROR] Nie udało się usunąć pliku '{file['name']}' z Firebase")


                if file in file_list:
                    file_list.remove(file)


            self.page.dialog.open = False
            self.page.update()


            if list_name == "local":
                self._save_local_files()


            self._refresh_containers()
            self.update()

        def on_cancel(e):

            self.page.dialog.open = False
            self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text(lang.t('kb_delete_confirm_title')),
            content=ft.Text(lang.t('kb_delete_confirm_message')),
            actions=[
                ft.TextButton(
                    lang.t('kb_delete_confirm_no'),
                    on_click=on_cancel,
                ),
                ft.TextButton(
                    lang.t('kb_delete_confirm_yes'),
                    on_click=on_confirm,
                ),
            ],
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def _on_refresh_clicked(self, list_name: str):

        if list_name == "firebase":

            if self.firebase.is_logged_in():
                firebase_file_list = self.firebase.list_user_files()
                current_user = self.firebase.get_current_user()
                username = current_user['username'] if current_user else "Unknown"


                self.firebase_files = []
                for fb_file in firebase_file_list:
                    file_size = fb_file.get('size', 0)
                    size_mb = f"{file_size / (1024 * 1024):.1f} MB"


                    filename = fb_file.get('filename', 'unknown')
                    import os
                    ext = os.path.splitext(filename)[1]


                    uploaded_at = fb_file.get('uploaded_at')
                    if uploaded_at:

                        import datetime
                        date_added = uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(uploaded_at, 'strftime') else str(uploaded_at)
                    else:
                        date_added = "-"

                    self.firebase_files.append({
                        "name": filename,
                        "location": f"Firebase - {username}",
                        "path": fb_file.get('path', ''),
                        "size": size_mb,
                        "ext": ext,
                        "attrs": "?",
                        "rows": "?",
                        "date_added": date_added,
                        "selected": False,
                        "firebase_id": fb_file.get('id'),
                    })

                print(f"[OK] Odświeżono listę plików Firebase: {len(self.firebase_files)} plików")
            else:
                print("[INFO] Użytkownik nie jest zalogowany - brak dostępu do Firebase")

        elif list_name == "local":

            print("[INFO] Odświeżono listę lokalnych plików")


        self._refresh_containers()
        self.update()

    def _on_new_experiment_clicked(self, file_list: list):





        selected_files = [f for f in file_list if f.get("selected", False)]

        if not selected_files:
            print("[ERROR] Brak zaznaczonych plików")
            return


        selected_file = selected_files[0]
        file_path = selected_file.get('path', '')

        if not file_path:
            print("[ERROR] Plik nie ma ścieżki")
            return

        print(f"[KB] Przechodzę do Nowego Eksperymentu z plikiem: {file_path}")


        if self.on_navigate:
            self.on_navigate(0, file_to_load=file_path)
        else:
            print("[ERROR] Brak callbacku on_navigate")

    def _create_firebase_section(self):


        if not self.firebase.is_logged_in():

            return ft.Column([
                ft.Text(
                    lang.t('kb_firebase_files'),
                    size=16,
                    weight=ft.FontWeight.W_600,
                    color=AppColors.TEXT_PRIMARY
                ),
                ft.Container(height=10),
                ft.Container(
                    content=ft.Text(
                        lang.t('kb_login_to_see_files'),
                        color=AppColors.TEXT_SECONDARY,
                        size=14,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    padding=ft.padding.all(20),
                    alignment=ft.alignment.center,
                ),
            ])
        else:

            return self._create_data_table_section(
                title=lang.t('kb_firebase_files'),
                file_list=self.firebase_files,
                list_name="firebase",
                show_checkboxes=True,
                show_delete_button=True,
            )

    def _create_data_table_section(self, title: str, file_list: list, list_name: str,
                                    show_checkboxes: bool = True, show_delete_button: bool = True):



        filtered_files = self._filter_files(file_list)


        action_buttons = []


        if list_name in ["local", "firebase"]:

            selected_count = sum(f.get("selected", False) for f in file_list)
            action_buttons.append(
                ft.ElevatedButton(
                    "Nowy eksperyment",
                    icon=ft.icons.PLAY_ARROW_ROUNDED,
                    on_click=lambda _: self._on_new_experiment_clicked(file_list),
                    bgcolor=AppColors.SECONDARY,
                    color=AppColors.TEXT_PRIMARY,
                    disabled=selected_count != 1,
                )
            )


        if list_name in ["local", "firebase"]:
            action_buttons.append(
                ft.ElevatedButton(
                    lang.t('kb_add_new'),
                    icon=ft.icons.ADD,
                    on_click=lambda _: self.file_picker.pick_files(
                        allowed_extensions=["csv"],
                        dialog_title=lang.t('kb_add_new'),
                    ),
                    bgcolor=AppColors.SECONDARY,
                    color=AppColors.TEXT_PRIMARY,
                )
            )

        if show_delete_button:
            action_buttons.append(
                ft.ElevatedButton(
                    lang.t('kb_delete'),
                    icon=ft.icons.DELETE_OUTLINE,
                    on_click=lambda _: self._on_delete_clicked(file_list, list_name),
                    bgcolor=AppColors.ERROR,
                    color=AppColors.TEXT_PRIMARY,
                )
            )

        action_buttons.append(
            ft.ElevatedButton(
                lang.t('kb_refresh'),
                icon=ft.icons.REFRESH,
                on_click=lambda _: self._on_refresh_clicked(list_name),
                bgcolor=AppColors.PRIMARY,
                color=AppColors.TEXT_PRIMARY,
            )
        )


        columns = []

        if show_checkboxes:
            columns.append(ft.DataColumn(ft.Text("", size=12)))

        columns.extend([
            ft.DataColumn(ft.Text(lang.t('kb_title_col'), color=AppColors.TEXT_SECONDARY, size=12)),
            ft.DataColumn(ft.Text(lang.t('kb_source_col'), color=AppColors.TEXT_SECONDARY, size=12)),
            ft.DataColumn(ft.Text(lang.t('kb_size_col'), color=AppColors.TEXT_SECONDARY, size=12)),
            ft.DataColumn(ft.Text(lang.t('kb_extension_col'), color=AppColors.TEXT_SECONDARY, size=12)),
            ft.DataColumn(ft.Text(lang.t('kb_attributes_col'), color=AppColors.TEXT_SECONDARY, size=12)),
            ft.DataColumn(ft.Text(lang.t('kb_rows_col'), color=AppColors.TEXT_SECONDARY, size=12)),
            ft.DataColumn(ft.Text(lang.t('kb_date_added'), color=AppColors.TEXT_SECONDARY, size=12)),
        ])


        rows = []
        for i, file_data in enumerate(filtered_files):
            cells = []


            if show_checkboxes:
                original_index = file_list.index(file_data)
                cells.append(
                    ft.DataCell(
                        ft.Checkbox(
                            value=file_data.get("selected", False),
                            on_change=lambda e, idx=original_index: self._on_checkbox_change(file_list, idx, e.control.value)
                        )
                    )
                )


            cells.extend([
                ft.DataCell(ft.Text(file_data["name"], color=AppColors.TEXT_PRIMARY, size=13, weight=ft.FontWeight.W_600)),
                ft.DataCell(ft.Text(file_data.get("location", file_data.get("path", "")), color=AppColors.TEXT_SECONDARY, size=12)),
                ft.DataCell(ft.Text(file_data["size"], color=AppColors.TEXT_PRIMARY, size=13)),
                ft.DataCell(ft.Text(file_data["ext"], color=AppColors.TEXT_PRIMARY, size=13)),
                ft.DataCell(ft.Text(str(file_data["attrs"]), color=AppColors.TEXT_PRIMARY, size=13)),
                ft.DataCell(ft.Text(str(file_data["rows"]), color=AppColors.TEXT_PRIMARY, size=13)),
                ft.DataCell(ft.Text(file_data.get("date_added", "-"), color=AppColors.TEXT_SECONDARY, size=12)),
            ])

            rows.append(ft.DataRow(cells=cells))


        return ft.Container(
            content=ft.Column([

                ft.Row([
                    ft.Text(title, size=16, weight=ft.FontWeight.W_600, color=AppColors.TEXT_PRIMARY),
                    ft.Row(action_buttons, spacing=10),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                ft.Container(height=10),


                ft.DataTable(
                    columns=columns,
                    rows=rows,
                    border_radius=8,
                    data_row_min_height=50,
                    heading_row_height=40,
                ),
            ]),
            padding=ft.padding.only(bottom=20),
        )

    def _create_rule_item(self, rule_id: str, premises: list, conclusion: str, cf: float):
        cf_color = AppColors.SECONDARY if cf >= 0.8 else (AppColors.WARNING if cf >= 0.6 else AppColors.ERROR)
        
        return ft.Container(
            content=ft.Row([
                ft.Text(rule_id, size=13, color=AppColors.PRIMARY, 
                       weight=ft.FontWeight.W_600, width=60),
                ft.Text("IF", size=12, color=AppColors.TEXT_MUTED, width=25),
                ft.Row([
                    create_chip(p, AppColors.TEXT_SECONDARY) for p in premises
                ], spacing=6),
                ft.Text("THEN", size=12, color=AppColors.TEXT_MUTED),
                create_chip(conclusion, AppColors.SECONDARY),
                ft.Container(expand=True),
                ft.Container(
                    content=ft.Text(f"CF={cf:.2f}", size=12, 
                                   color=cf_color, weight=ft.FontWeight.W_600),
                    padding=ft.padding.symmetric(horizontal=10, vertical=4),
                    border_radius=6,
                    bgcolor=ft.colors.with_opacity(0.15, cf_color),
                ),
            ], spacing=8),
            padding=ft.padding.symmetric(vertical=12, horizontal=4),
            border=ft.border.only(bottom=ft.BorderSide(1, AppColors.BORDER)),
        )



class ExperimentDetailView(ft.UserControl):



    def __init__(self, on_close=None):
        super().__init__()
        self.on_close = on_close
        self.storage = None
        self.experiment_dir = None
        self.metadata = None


        self.metrics_container = ft.Container()
        self.config_container = ft.Container()
        self.logs_container = ft.Container()

    def load_experiment(self, experiment_dir):

        from pathlib import Path
        from core.storage import ExperimentStorage

        self.experiment_dir = Path(experiment_dir) if isinstance(experiment_dir, str) else experiment_dir
        self.storage = ExperimentStorage()
        self.metadata = self.storage.load_experiment_metadata(self.experiment_dir)

        if self.metadata:
            self._build_metrics()
            self._build_config()
            self._load_logs_async()
            self.update()

    def _build_metrics(self):

        metrics = self.metadata.get('metrics', {})

        execution_time = metrics.get('execution_time_ms', 0)
        rules_count = metrics.get('rules_count', 0)
        new_facts = metrics.get('new_facts_count', 0)
        iterations = metrics.get('iterations', 0)

        self.metrics_container.content = ft.Column([
            create_section_header("📊 Metryki Eksperymentu"),
            ft.Container(height=16),
            ft.Row([
                create_stat_card("Czas wykonania", f"{execution_time:.2f} ms",
                               ft.icons.TIMER_ROUNDED, AppColors.PRIMARY),
                create_stat_card("Liczba reguł", str(rules_count),
                               ft.icons.RULE_ROUNDED, AppColors.SECONDARY),
                create_stat_card("Nowe fakty", str(new_facts),
                               ft.icons.LIGHTBULB_ROUNDED, AppColors.WARNING),
                create_stat_card("Iteracje", str(iterations),
                               ft.icons.REPEAT_ROUNDED, AppColors.TEXT_MUTED),
            ], spacing=16),
        ])

    def _build_config(self):

        config = self.metadata.get('config', {})

        config_items = [
            ("Seed", config.get('seed')),
            ("Strategia", config.get('strategy')),
            ("Metoda generowania", config.get('generate_method')),
            ("Metoda wnioskowania", config.get('inference_method')),
            ("Kolumna decyzyjna", config.get('decision_column')),
            ("Dyskretyzacja", f"{config.get('discretization_method')} ({config.get('discretization_bins')} bins)"),
        ]


        if config.get('generate_method') == 'Tree':
            config_items.extend([
                ("Tree max_depth", config.get('tree_max_depth')),
                ("Tree min_samples_leaf", config.get('tree_min_samples_leaf')),
            ])
        elif config.get('generate_method') == 'Forest':
            config_items.extend([
                ("Forest n_estimators", config.get('forest_n_estimators')),
                ("Forest min_depth", config.get('forest_min_depth')),
                ("Forest max_depth", config.get('forest_max_depth')),
                ("Forest min_samples_leaf", config.get('forest_min_samples_leaf')),
            ])

        if config.get('clustering_enabled'):
            config_items.append(("Klasteryzacja", f"Włączona ({config.get('n_clusters')} klastrów)"))

        config_rows = []
        for label, value in config_items:
            config_rows.append(
                ft.Row([
                    ft.Text(f"{label}:", size=14, color=AppColors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                    ft.Text(str(value), size=14, color=AppColors.TEXT_PRIMARY),
                ], spacing=8)
            )

        self.config_container.content = ft.Column([
            create_section_header("⚙️ Konfiguracja Eksperymentu"),
            ft.Container(height=16),
            create_card(ft.Column(config_rows, spacing=8)),
        ])

    def _load_logs_async(self):

        import threading

        def load_logs():
            log_content = self.storage.load_log_file(self.experiment_dir, log_type="extended")

            if log_content:

                def update_ui():
                    self.logs_container.content = ft.Column([
                        create_section_header("📋 Logi XAI (Rozszerzone)"),
                        ft.Container(height=16),
                        create_card(
                            ft.TextField(
                                value=log_content,
                                multiline=True,
                                min_lines=15,
                                max_lines=20,
                                read_only=True,
                                text_size=12,
                                bgcolor=AppColors.BG_ELEVATED,
                                border_color=AppColors.BORDER,
                            )
                        ),
                    ])
                    self.update()

                if self.page:
                    self.page.run_task(update_ui)
            else:
                def update_ui():
                    self.logs_container.content = ft.Column([
                        create_section_header("📋 Logi XAI"),
                        ft.Container(height=16),
                        ft.Text("Brak pliku logu dla tego eksperymentu.",
                               color=AppColors.TEXT_SECONDARY, italic=True),
                    ])
                    self.update()

                if self.page:
                    self.page.run_task(update_ui)


        thread = threading.Thread(target=load_logs, daemon=True)
        thread.start()

    def build(self):
        if not self.metadata:
            return ft.Column([
                ft.Text("Ładowanie danych eksperymentu...",
                       size=16, color=AppColors.TEXT_SECONDARY)
            ])


        header = ft.Row([
            ft.IconButton(
                icon=ft.icons.ARROW_BACK_ROUNDED,
                tooltip="Powrót do listy",
                on_click=lambda _: self.on_close() if self.on_close else None,
            ),
            ft.Column([
                ft.Text(f"Eksperyment: {self.metadata.get('run_id')}",
                       size=24, weight=ft.FontWeight.BOLD, color=AppColors.TEXT_PRIMARY),
                ft.Text(f"Dataset: {self.metadata.get('dataset_name')} | "
                       f"Timestamp: {self.metadata.get('timestamp')}",
                       size=14, color=AppColors.TEXT_SECONDARY),
            ], spacing=4),
        ])

        return ft.Column([
            header,
            ft.Container(height=20),
            self.metrics_container,
            ft.Container(height=20),
            self.config_container,
            ft.Container(height=20),
            self.logs_container,
        ], scroll=ft.ScrollMode.AUTO)


class HistoryView(ft.UserControl):



    def __init__(self, on_experiment_click=None):
        super().__init__()
        self.on_experiment_click = on_experiment_click
        self.storage = None
        self.experiments_list = ft.ListView(spacing=12, expand=True)

    def load_experiments(self):

        import os
        from pathlib import Path


        results_path = Path(__file__).parent.parent / "results"

        self.experiments_list.controls.clear()


        benchmark_dirs = []
        if results_path.exists():
            for item in results_path.iterdir():
                if item.is_dir() and item.name.startswith("benchmark_"):
                    benchmark_dirs.append(item)


        benchmark_dirs.sort(key=lambda x: x.name, reverse=True)

        if not benchmark_dirs:
            self.experiments_list.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.FOLDER_OFF_ROUNDED, size=48, color=AppColors.TEXT_MUTED),
                        ft.Text("Brak benchmarków", size=16, color=AppColors.TEXT_SECONDARY),
                        ft.Text("Uruchom benchmark w zakładce 'Nowy Eksperyment'",
                               size=12, color=AppColors.TEXT_MUTED),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
                    padding=40,
                    alignment=ft.alignment.center,
                )
            )
        else:
            for benchmark_dir in benchmark_dirs:
                self.experiments_list.controls.append(
                    self._create_benchmark_card(benchmark_dir)
                )

        self.update()

    def _create_benchmark_card(self, benchmark_path):

        import os
        import subprocess
        import platform
        from datetime import datetime


        folder_name = benchmark_path.name
        try:
            timestamp_str = folder_name.replace("benchmark_", "")
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            date_display = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_display = folder_name


        traces_path = benchmark_path / "traces"
        trace_count = 0
        if traces_path.exists():
            trace_count = len([f for f in traces_path.iterdir() if f.is_file()])


        csv_files = list(benchmark_path.glob("*.csv"))
        has_csv = len(csv_files) > 0


        def open_folder(_):
            folder_path = str(benchmark_path)
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", folder_path])
            else:
                subprocess.run(["xdg-open", folder_path])

        card = ft.Container(
            content=ft.Row([

                ft.Container(
                    content=ft.Icon(ft.icons.SCIENCE_ROUNDED, size=32, color=AppColors.SECONDARY),
                    width=60,
                    alignment=ft.alignment.center,
                ),

                ft.Column([
                    ft.Row([
                        ft.Text("Benchmark", size=16, weight=ft.FontWeight.BOLD,
                               color=AppColors.TEXT_PRIMARY),
                        ft.Container(
                            content=ft.Text(f"{trace_count} prób", size=12, color=ft.colors.WHITE),
                            bgcolor=AppColors.SECONDARY,
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            border_radius=4,
                        ),
                        ft.Container(
                            content=ft.Text("CSV", size=10, color=ft.colors.WHITE),
                            bgcolor=AppColors.PRIMARY if has_csv else AppColors.TEXT_MUTED,
                            padding=ft.padding.symmetric(horizontal=6, vertical=2),
                            border_radius=4,
                        ) if has_csv else ft.Container(),
                    ], spacing=12),
                    ft.Text(f"Data: {date_display}",
                           size=12, color=AppColors.TEXT_SECONDARY),
                    ft.Text(f"Folder: {folder_name}",
                           size=11, color=AppColors.TEXT_MUTED),
                ], spacing=4, expand=True),

                ft.ElevatedButton(
                    "Otwórz folder",
                    icon=ft.icons.FOLDER_OPEN_ROUNDED,
                    on_click=open_folder,
                    bgcolor=AppColors.BG_ELEVATED,
                    color=AppColors.TEXT_PRIMARY,
                ),
            ], spacing=16),
            padding=16,
            border_radius=12,
            bgcolor=AppColors.BG_CARD,
            border=ft.border.all(1, AppColors.BORDER),
        )

        return card

    def _create_experiment_card(self, summary):


        status_icon = "💻" if summary['sync_status'] == 'local' else "☁️"


        success_color = AppColors.SECONDARY if summary.get('success', False) else AppColors.ERROR
        success_text = "Sukces" if summary['success'] else "Porażka"


        timestamp = summary.get('timestamp', '')
        if timestamp:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_str = timestamp
        else:
            timestamp_str = "N/A"

        card = ft.Container(
            content=ft.Row([

                ft.Container(
                    content=ft.Text(status_icon, size=32),
                    width=60,
                    alignment=ft.alignment.center,
                ),

                ft.Column([
                    ft.Row([
                        ft.Text(summary['dataset_name'], size=16, weight=ft.FontWeight.BOLD,
                               color=AppColors.TEXT_PRIMARY),
                        ft.Container(
                            content=ft.Text(success_text, size=12, color=ft.colors.WHITE),
                            bgcolor=success_color,
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            border_radius=4,
                        ),
                    ], spacing=12),
                    ft.Text(f"Metoda: {summary['generate_method']} | Strategia: {summary['strategy']}",
                           size=12, color=AppColors.TEXT_SECONDARY),
                    ft.Text(f"Data: {timestamp_str}",
                           size=12, color=AppColors.TEXT_MUTED),
                ], spacing=4, expand=True),

                ft.Column([
                    ft.Text(f"{summary['execution_time_ms']:.2f} ms",
                           size=12, color=AppColors.PRIMARY, weight=ft.FontWeight.BOLD),
                    ft.Text(f"{summary['rules_count']} reguł",
                           size=11, color=AppColors.TEXT_SECONDARY),
                    ft.Text(f"{summary['new_facts_count']} faktów",
                           size=11, color=AppColors.TEXT_SECONDARY),
                ], horizontal_alignment=ft.CrossAxisAlignment.END, spacing=2),
            ], spacing=16),
            padding=16,
            border_radius=12,
            bgcolor=AppColors.BG_CARD,
            border=ft.border.all(1, AppColors.BORDER),
            ink=True,
            on_click=lambda _, s=summary: self._on_card_click(s),
        )

        return card

    def _on_card_click(self, summary):

        if self.on_experiment_click:
            self.on_experiment_click(summary['folder_path'])

    def did_mount(self):


        import threading
        import time

        def load_async():
            time.sleep(0.1)
            try:
                self.load_experiments()
            except Exception as e:
                print(f"[ERROR] Błąd podczas ładowania eksperymentów: {e}")

        thread = threading.Thread(target=load_async, daemon=True)
        thread.start()

    def build(self):
        return ft.Column([

            ft.Row([
                ft.Column([
                    ft.Text("Wyniki Benchmarków", size=28, weight=ft.FontWeight.BOLD,
                           color=AppColors.TEXT_PRIMARY),
                    ft.Text("Przeglądaj wyniki przeprowadzonych benchmarków", size=14,
                           color=AppColors.TEXT_SECONDARY),
                ], spacing=4),
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.icons.REFRESH_ROUNDED,
                    tooltip="Odśwież listę",
                    on_click=lambda _: self.load_experiments(),
                ),
            ]),

            ft.Container(height=20),


            ft.Container(
                content=self.experiments_list,
                expand=True,
            ),
        ], expand=True)


class SettingsView(ft.UserControl):
    def __init__(self, state_manager=None, session_expired_message=None):
        super().__init__()


        self.firebase = FirebaseService()


        self.state_manager = state_manager if state_manager else AppStateManager()


        self.session_expired_message = session_expired_message


        self.is_registering = False


        self.login_username = ""
        self.login_password = ""
        self.status_message = ""
        self.status_color = AppColors.TEXT_MUTED

    def _handle_language_change(self, e):

        new_lang = e.control.value
        lang.set_language(new_lang)

        if hasattr(self.page, 'on_language_changed'):
            self.page.on_language_changed()

    def _handle_keep_logged_in_change(self, e):

        new_value = e.control.value


        if self.firebase.is_logged_in():
            user_data = self.firebase.get_current_user()
            username = user_data['username']
            user_id = user_data['id']


            self.state_manager.set_logged_in(username, user_id, new_value)

            print(f"[SETTINGS] Zmieniono 'Nie wylogowuj mnie' na: {new_value}")
        else:

            print(f"[SETTINGS] Przełącznik ustawiony na: {new_value} (zapisze się po zalogowaniu)")

    def _handle_login(self, e):

        username = self.username_field.value
        password = self.password_field.value if self.password_field.value else ""

        if not username:
            self.status_text.value = "Login jest wymagany"
            self.status_text.color = AppColors.WARNING
            self.update()
            return


        if self.firebase.login(username, password):

            self.status_text.value = ""
            self.current_user_text.value = f"{lang.t('settings_logged_in')} {username}"
            self.current_user_text.visible = True
            self.login_container.visible = False
            self.logout_button.visible = True
            self.keep_logged_in_row.visible = True


            user_id = self.firebase.current_user['id']
            keep_logged_in = self.keep_logged_in_switch.value
            self.state_manager.set_logged_in(username, user_id, keep_logged_in)

            print(f"[LOGIN] Zapisano stan logowania (keep_logged_in={keep_logged_in})")


            if hasattr(self.page, 'sidebar') and self.page.sidebar:
                self.page.sidebar.refresh_user_info()
        else:
            self.status_text.value = lang.t('settings_login_failed')
            self.status_text.color = AppColors.ERROR

        self.update()

    def _handle_logout(self, e):

        self.firebase.logout()


        self.state_manager.clear_login()
        print("[LOGOUT] Wyczyszczono stan logowania")

        self.status_text.value = lang.t('settings_logout_success')
        self.status_text.color = AppColors.SECONDARY
        self.current_user_text.visible = False
        self.login_container.visible = True
        self.logout_button.visible = False
        self.keep_logged_in_row.visible = True
        self.username_field.value = ""
        self.password_field.value = ""


        self.keep_logged_in_switch.value = False


        if hasattr(self.page, 'sidebar') and self.page.sidebar:
            self.page.sidebar.refresh_user_info()

        self.update()

    def _handle_show_registration(self, e):

        self.is_registering = True

        self._rebuild_firebase_section()

    def _handle_cancel_registration(self, e):

        self.is_registering = False
        self.status_text.value = ""

        self._rebuild_firebase_section()

    def _handle_confirm_registration(self, e):

        name = self.register_name_field.value
        username = self.register_username_field.value
        password = self.register_password_field.value
        password_repeat = self.register_password_repeat_field.value


        if not name or not username:
            self.status_text.value = lang.t('settings_register_fill_required')
            self.status_text.color = AppColors.ERROR
            self.update()
            return


        if password or password_repeat:
            if password != password_repeat:
                self.status_text.value = lang.t('settings_register_passwords_mismatch')
                self.status_text.color = AppColors.ERROR
                self.update()
                return


        if not password:
            password = ""


        success = self.firebase.create_user(username, password)

        if success:
            self.status_text.value = lang.t('settings_register_success')
            self.status_text.color = AppColors.SECONDARY
            self.is_registering = False

            self._rebuild_firebase_section()
        else:
            self.status_text.value = lang.t('settings_user_create_failed')
            self.status_text.color = AppColors.ERROR
            self.update()

    def _rebuild_firebase_section(self):


        if self.is_registering:

            self.login_container.visible = False
            self.register_container.visible = True
            self.keep_logged_in_row.visible = False
        else:

            self.login_container.visible = True
            self.register_container.visible = False
            self.keep_logged_in_row.visible = True

        self.update()

    def build(self):

        self.username_field = ft.TextField(
            label=lang.t('settings_login'),
            hint_text=lang.t('settings_login_placeholder'),
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )

        self.password_field = ft.TextField(
            label=lang.t('settings_password'),
            hint_text=lang.t('settings_password_placeholder'),
            password=True,
            can_reveal_password=True,
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )


        self.register_name_field = ft.TextField(
            label=lang.t('settings_register_name'),
            hint_text=lang.t('settings_register_name_placeholder'),
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )

        self.register_username_field = ft.TextField(
            label=lang.t('settings_register_username'),
            hint_text=lang.t('settings_register_username_placeholder'),
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )

        self.register_password_field = ft.TextField(
            label=lang.t('settings_register_password_optional'),
            hint_text=lang.t('settings_register_password_placeholder'),
            password=True,
            can_reveal_password=True,
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )

        self.register_password_repeat_field = ft.TextField(
            label=lang.t('settings_register_password_repeat'),
            hint_text=lang.t('settings_register_password_repeat_placeholder'),
            password=True,
            can_reveal_password=True,
            border_color=AppColors.BORDER,
            focused_border_color=AppColors.PRIMARY,
            width=250,
        )

        self.status_text = ft.Text(
            "",
            size=13,
            color=AppColors.TEXT_MUTED,
            weight=ft.FontWeight.W_500,
        )

        self.current_user_text = ft.Text(
            "",
            size=14,
            color=AppColors.SECONDARY,
            weight=ft.FontWeight.BOLD,
            visible=False,
        )

        self.login_container = ft.Column([
            ft.Row([
                self.username_field,
                self.password_field,
            ], spacing=16),
            ft.Container(height=12),
            ft.Row([
                ft.ElevatedButton(
                    lang.t('settings_login_button'),
                    icon=ft.icons.LOGIN_ROUNDED,
                    style=ft.ButtonStyle(
                        bgcolor=AppColors.PRIMARY,
                        color=AppColors.TEXT_PRIMARY,
                    ),
                    on_click=self._handle_login,
                ),
                ft.OutlinedButton(
                    lang.t('settings_create_user'),
                    icon=ft.icons.PERSON_ADD_ROUNDED,
                    style=ft.ButtonStyle(
                        color=AppColors.SECONDARY,
                        side=ft.BorderSide(1, AppColors.SECONDARY),
                    ),
                    on_click=self._handle_show_registration,
                ),
            ], spacing=12),
        ], spacing=8, visible=True)

        self.register_container = ft.Column([
            ft.Text(
                lang.t('settings_register_title'),
                size=16,
                color=AppColors.TEXT_PRIMARY,
                weight=ft.FontWeight.BOLD,
            ),
            ft.Container(height=12),
            ft.Row([
                self.register_name_field,
                self.register_username_field,
            ], spacing=16),
            ft.Container(height=12),
            ft.Row([
                self.register_password_field,
                self.register_password_repeat_field,
            ], spacing=16),
            ft.Container(height=8),
            ft.Text(
                lang.t('settings_register_password_info'),
                size=12,
                color=AppColors.TEXT_MUTED,
                italic=True,
            ),
            ft.Container(height=12),
            ft.Row([
                ft.ElevatedButton(
                    lang.t('settings_register_confirm'),
                    icon=ft.icons.CHECK_ROUNDED,
                    style=ft.ButtonStyle(
                        bgcolor=AppColors.SECONDARY,
                        color=AppColors.TEXT_PRIMARY,
                    ),
                    on_click=self._handle_confirm_registration,
                ),
                ft.OutlinedButton(
                    lang.t('settings_register_cancel'),
                    icon=ft.icons.CLOSE_ROUNDED,
                    style=ft.ButtonStyle(
                        color=AppColors.ERROR,
                        side=ft.BorderSide(1, AppColors.ERROR),
                    ),
                    on_click=self._handle_cancel_registration,
                ),
            ], spacing=12),
        ], spacing=8, visible=False)

        self.logout_button = ft.ElevatedButton(
            lang.t('settings_logout'),
            icon=ft.icons.LOGOUT_ROUNDED,
            style=ft.ButtonStyle(
                bgcolor=AppColors.ERROR,
                color=AppColors.TEXT_PRIMARY,
            ),
            on_click=self._handle_logout,
            visible=False,
        )


        self.keep_logged_in_switch = ft.Switch(
            value=False,
            active_color=AppColors.PRIMARY,
            on_change=self._handle_keep_logged_in_change,
        )


        self.keep_logged_in_row = ft.Row([
            self.keep_logged_in_switch,
            ft.Text(lang.t('settings_keep_logged_in'),
                   color=AppColors.TEXT_PRIMARY, size=14),
        ], spacing=12, visible=True)


        if self.firebase.is_logged_in():
            user_data = self.firebase.get_current_user()
            username = user_data['username']
            self.current_user_text.value = f"{lang.t('settings_logged_in')} {username}"
            self.current_user_text.visible = True
            self.login_container.visible = False
            self.logout_button.visible = True
            self.keep_logged_in_row.visible = True


            self.keep_logged_in_switch.value = self.state_manager.get_keep_logged_in()


        if self.session_expired_message:
            self.status_text.value = self.session_expired_message
            self.status_text.color = AppColors.WARNING

        return ft.Column([
            ft.Text(lang.t('settings_title'), size=28, weight=ft.FontWeight.BOLD,
                   color=AppColors.TEXT_PRIMARY),
            ft.Text(lang.t('settings_subtitle'), size=14,
                   color=AppColors.TEXT_SECONDARY),

            ft.Container(height=30),


            create_card(
                ft.Column([
                    create_section_header(lang.t('settings_general')),
                    ft.Container(height=16),


                    ft.Row([
                        ft.Text(lang.t('settings_language'), size=13,
                               color=AppColors.TEXT_SECONDARY, width=150),
                        ft.Dropdown(
                            value=lang.get_current_language(),
                            options=[
                                ft.dropdown.Option("pl", lang.t('settings_language_polish')),
                                ft.dropdown.Option("en", lang.t('settings_language_english')),
                            ],
                            on_change=self._handle_language_change,
                            border_color=AppColors.BORDER,
                            focused_border_color=AppColors.PRIMARY,
                            width=200,
                        ),
                    ], spacing=16),
                ])
            ),

            ft.Container(height=20),


            create_card(
                ft.Column([
                    create_section_header(lang.t('settings_firebase'),
                                        lang.t('settings_firebase_subtitle')),
                    ft.Container(height=8),


                    ft.Row([
                        ft.Icon(
                            ft.icons.CLOUD_DONE_ROUNDED if self.firebase.db else ft.icons.CLOUD_OFF_ROUNDED,
                            color=AppColors.SECONDARY if self.firebase.db else AppColors.ERROR,
                            size=20
                        ),
                        ft.Text(
                            lang.t('settings_connected') if self.firebase.db else lang.t('settings_disconnected'),
                            size=13,
                            color=AppColors.SECONDARY if self.firebase.db else AppColors.ERROR,
                            weight=ft.FontWeight.W_500
                        ),
                    ], spacing=8),

                    ft.Container(height=8),


                    self.current_user_text,


                    self.login_container,


                    self.register_container,


                    self.logout_button,

                    ft.Container(height=8),


                    self.keep_logged_in_row,

                    ft.Container(height=8),


                    self.status_text,
                ])
            ),

            ft.Container(height=20),


            create_card(
                ft.Column([
                    create_section_header(lang.t('settings_about')),
                    ft.Container(height=16),

                    ft.Column([
                        ft.Text(lang.t('settings_version'), size=16,
                               color=AppColors.TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
                        ft.Text(lang.t('settings_thesis'),
                               size=13, color=AppColors.TEXT_SECONDARY),
                        ft.Text(lang.t('settings_university'), size=12,
                               color=AppColors.TEXT_MUTED),
                    ], spacing=4),
                ])
            ),
        ], scroll=ft.ScrollMode.AUTO)


def main(page: ft.Page):
    page.title = "Expert System"
    page.bgcolor = AppColors.BG_DARK
    page.padding = 0
    page.window_width = 1400
    page.window_height = 900
    page.window_min_width = 1200
    page.window_min_height = 700
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            on_surface=AppColors.TEXT_PRIMARY,
            on_surface_variant=AppColors.TEXT_SECONDARY,
        ),
    )
    page.dark_theme = page.theme
    page.theme_mode = ft.ThemeMode.DARK



    state_manager = AppStateManager()
    firebase = FirebaseService()


    splash_container = ft.Container(
        content=ft.Column([
            ft.ProgressRing(color=AppColors.PRIMARY),
            ft.Container(height=20),
            ft.Text(
                "Przywracanie sesji...",
                size=18,
                color=AppColors.TEXT_PRIMARY,
                weight=ft.FontWeight.W_500
            )
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        alignment=ft.alignment.center,
        expand=True
    )


    auto_login_attempted = False
    session_expired_message = None

    if state_manager.get_keep_logged_in():
        last_user = state_manager.get_last_user()

        if last_user:
            print(f"[AUTO-LOGIN] Próba przywrócenia sesji dla: {last_user['username']}")


            page.add(splash_container)
            page.update()


            user_id = last_user['user_id']
            if firebase.validate_session(user_id):

                if firebase.auto_login(user_id):
                    print(f"[AUTO-LOGIN] Sukces! Zalogowano jako: {last_user['username']}")
                    auto_login_attempted = True
                else:
                    print("[AUTO-LOGIN] Błąd podczas auto-logowania")
                    session_expired_message = "Nie udało się przywrócić sesji"
                    state_manager.clear_login()
            else:

                print("[AUTO-LOGIN] Sesja wygasła")
                session_expired_message = "Sesja wygasła. Zaloguj się ponownie."
                state_manager.clear_login()


            page.clean()


    sidebar = None
    current_view_index = [-1]


    def on_language_changed():

        if sidebar:
            sidebar._update_menu_items()
            sidebar._update_menu()
            sidebar.update()


        if current_view_index[0] >= 0:
          
            def show_experiment_detail(experiment_dir):

                detail_view = ExperimentDetailView(
                    on_close=lambda: on_navigate(2, None)
                )
                detail_view.load_experiment(experiment_dir)
                content_container.content = detail_view
                page.update()

            new_views = [
                NewExperimentView(),
                KnowledgeBaseView(on_navigate=on_navigate),
                HistoryView(on_experiment_click=show_experiment_detail),
                SettingsView(state_manager=state_manager, session_expired_message=session_expired_message),
            ]
            content_container.content = new_views[current_view_index[0]]
            page.update()


    page.on_language_changed = on_language_changed


    def on_navigate(index: int, file_to_load: str = None):







        current_view_index[0] = index


        if sidebar:
            sidebar.selected_index = index
            sidebar._update_menu()
            sidebar.update()


      
        def show_experiment_detail(experiment_dir):

            detail_view = ExperimentDetailView(
                on_close=lambda: on_navigate(2, None)
            )
            detail_view.load_experiment(experiment_dir)
            content_container.content = detail_view
            page.update()


        if index == 0 and file_to_load:
            new_experiment_view = NewExperimentView()
            new_experiment_view.preload_file_path = file_to_load
            new_views = [
                new_experiment_view,
                KnowledgeBaseView(on_navigate=on_navigate),
                HistoryView(on_experiment_click=show_experiment_detail),
                SettingsView(state_manager=state_manager, session_expired_message=session_expired_message),
            ]
        else:
            new_views = [
                NewExperimentView(),
                KnowledgeBaseView(on_navigate=on_navigate),
                HistoryView(on_experiment_click=show_experiment_detail),
                SettingsView(state_manager=state_manager, session_expired_message=session_expired_message),
            ]
        content_container.content = new_views[index]
        page.update()


    initial_view = ft.Container()


    content_container = ft.Container(
        content=initial_view,
        expand=True,
        padding=30,
    )


    sidebar = Sidebar(on_navigate)


    page.sidebar = sidebar


    page.add(
        ft.Row([
            sidebar,
            content_container,
        ], expand=True, spacing=0)
    )


if __name__ == "__main__":
    ft.app(target=main)
