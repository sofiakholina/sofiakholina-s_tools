# s_tools

**Русский** · [English](#english)

Ноды ComfyUI для комфортной работы: сохранения 3D с текстурой, FBX на входе, предпросмотр моделей в режиме Clay и другое. Устанавливаются автоматически установщиком из репозиториев:

- [sofiakholina-3D-Generation](https://github.com/sofiakholina/sofiakholina-3D-Generation) — генерация 3D-модели (без текстуры) по одному или нескольким видам в ComfyUI;
- [sofiakholina-3D-Texturing](https://github.com/sofiakholina/sofiakholina-3D-Texturing) — генерация текстуры для своей 3D-модели по одному или нескольким видам в ComfyUI.

## Ноды

| Нода | Что делает |
|---|---|
| **Save 3D** | Сохраняет модель в FBX / OBJ / GLB, с текстурой (PNG рядом с FBX/OBJ, внутри GLB). Размер модели сохраняется реальным в любом формате. Кнопки Save (сохранить последний результат без перезапуска) и Choose folder. |
| **Mesh Clay** | Серый материал «clay» для просмотра 3D без цветов. |
| **Load Image (optional)** | Загрузка изображения, которую можно оставить пустой: пустой вид пропускается. |
| **UltraTex Mesh** / **UltraTex Mesh (GGUF)** | Модель для UltraTex по пути к файлу: OBJ, FBX, GLB, PLY, STL; своя UV или автоматическая развёртка. |
| **UltraTex Bake 6/10** | Запекание UltraTex с выбором 6 или 10 видов. |
| **UltraTex View Gray / Side From Front/Back / Photo Views** | Текстурирование по нескольким фото (перед, спина, бока). |
| **UltraTex View Count (GGUF)**, **UltraTex Cache -> temp** | Служебные ноды для GGUF-варианта. |

### Единицы в Save 3D

FBX и GLB хранят реальный размер сами. У OBJ единиц нет, поэтому OBJ записывается в единицах сцены: скрытая настройка `save_units` (по умолчанию mm) должна совпадать с единицами сцены 3ds Max (Customize → Units Setup → System Unit Setup). В Nodes 2.0 скрытые настройки открываются кнопкой «Показать расширенные входные данные».

## Ручная установка

1. Скачайте архив [main.zip](https://github.com/sofiakholina/sofiakholina-s_tools/archive/refs/heads/main.zip) (или зелёная кнопка **Code** вверху страницы → **Download ZIP**), распакуйте, папку `sofiakholina-s_tools-main` переименуйте в `s_tools` и положите в папку нод (Comfy Desktop: `C:\Users\<имя пользователя>\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes`, Portable: `<папка, куда вы распаковали Portable>\ComfyUI_windows_portable\ComfyUI\custom_nodes`) — важно: внутри папки s_tools сразу должен быть набор файлов, а не еще одна папка.
2. Перезапустите ComfyUI.

Дополнительных пакетов не требуется. Ноды UltraTex работают вместе с [ComfyUI-UltraTex](https://github.com/visualbruno/ComfyUI-UltraTex) (и [UltraTex-ComfyUI-nodes](https://github.com/Mstafa-awad/UltraTex-ComfyUI-nodes) для GGUF) — их ставит установщик workflow.

## Лицензия

GPL-3.0 (см. [LICENSE](LICENSE)). Ноды s_tools для GGUF-варианта работают вместе с нодами MostAadTech (GPL-3.0).

---

<a name="english"></a>
## English

ComfyUI nodes for a comfortable workflow: saving 3D with textures, FBX input, Clay preview of models and more. Installed automatically by the installers in:

- [sofiakholina-3D-Generation](https://github.com/sofiakholina/sofiakholina-3D-Generation) — 3D model generation (untextured) from one or several views in ComfyUI;
- [sofiakholina-3D-Texturing](https://github.com/sofiakholina/sofiakholina-3D-Texturing) — texture generation for your own 3D model from one or several views in ComfyUI.

### Nodes

| Node | What it does |
|---|---|
| **Save 3D** | Saves a model as FBX / OBJ / GLB with its texture (PNG next to FBX/OBJ, embedded in GLB). Real-world size is kept in every format. Save button (saves the last result without re-running) and Choose folder. |
| **Mesh Clay** | Grey "clay" material for previewing 3D without colours. |
| **Load Image (optional)** | Image upload that may stay empty: an empty view is skipped. |
| **UltraTex Mesh** / **UltraTex Mesh (GGUF)** | Mesh for UltraTex from a file path: OBJ, FBX, GLB, PLY, STL; own UVs or automatic unwrap. |
| **UltraTex Bake 6/10** | UltraTex bake with a 6 or 10 view choice. |
| **UltraTex View Gray / Side From Front/Back / Photo Views** | Texturing from several photos (front, back, sides). |
| **UltraTex View Count (GGUF)**, **UltraTex Cache -> temp** | Helper nodes for the GGUF version. |

#### Units in Save 3D

FBX and GLB store the real size themselves. OBJ has no units, so OBJ is written in your scene units: the advanced setting `save_units` (default mm) must match your 3ds Max scene units (Customize → Units Setup → System Unit Setup). In Nodes 2.0, advanced settings are shown with the "Show advanced inputs" button.

### Manual install

1. Download [main.zip](https://github.com/sofiakholina/sofiakholina-s_tools/archive/refs/heads/main.zip) (or the green **Code** button at the top of the page → **Download ZIP**), unpack it, rename the `sofiakholina-s_tools-main` folder to `s_tools` and put it into the nodes folder (Comfy Desktop: `C:\Users\<user name>\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes`, Portable: `<folder where you unpacked Portable>\ComfyUI_windows_portable\ComfyUI\custom_nodes`) — important: the files must be directly inside the s_tools folder, not inside another nested folder.
2. Restart ComfyUI.

No extra packages are needed. The UltraTex nodes work together with [ComfyUI-UltraTex](https://github.com/visualbruno/ComfyUI-UltraTex) (and [UltraTex-ComfyUI-nodes](https://github.com/Mstafa-awad/UltraTex-ComfyUI-nodes) for GGUF), installed by the workflow installer.

### License

GPL-3.0 (see [LICENSE](LICENSE)). The GGUF helper nodes work together with MostAadTech's nodes (GPL-3.0).
