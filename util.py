import streamlit as st
import spacy
import base64


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model."""
    return spacy.load(name)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def process_text(model_name: str, text: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object."""
    nlp = load_model(model_name)
    return nlp(text)


def get_svg(svg: str, style: str = "", wrap: bool = True):
    """Convert an SVG to a base64-encoded image."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'
    return get_html(html) if wrap else html


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


# LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 900 500 175" width="150" height="53"><path fill="#09A3D5" d="M64.8 970.6c-11.3-1.3-12.2-16.5-26.7-15.2-7 0-13.6 2.9-13.6 9.4 0 9.7 15 10.6 24.1 13.1 15.4 4.7 30.4 7.9 30.4 24.7 0 21.3-16.7 28.7-38.7 28.7-18.4 0-37.1-6.5-37.1-23.5 0-4.7 4.5-8.4 8.9-8.4 5.5 0 7.5 2.3 9.4 6.2 4.3 7.5 9.1 11.6 21 11.6 7.5 0 15.3-2.9 15.3-9.4 0-9.3-9.5-11.3-19.3-13.6-17.4-4.9-32.3-7.4-34-26.7-1.8-32.9 66.7-34.1 70.6-5.3-.3 5.2-5.2 8.4-10.3 8.4zm81.5-28.8c24.1 0 37.7 20.1 37.7 44.9 0 24.9-13.2 44.9-37.7 44.9-13.6 0-22.1-5.8-28.2-14.7v32.9c0 9.9-3.2 14.7-10.4 14.7-8.8 0-10.4-5.6-10.4-14.7v-95.6c0-7.8 3.3-12.6 10.4-12.6 6.7 0 10.4 5.3 10.4 12.6v2.7c6.8-8.5 14.6-15.1 28.2-15.1zm-5.7 72.8c14.1 0 20.4-13 20.4-28.2 0-14.8-6.4-28.2-20.4-28.2-14.7 0-21.5 12.1-21.5 28.2.1 15.7 6.9 28.2 21.5 28.2zm59.8-49.3c0-17.3 19.9-23.5 39.2-23.5 27.1 0 38.2 7.9 38.2 34v25.2c0 6 3.7 17.9 3.7 21.5 0 5.5-5 8.9-10.4 8.9-6 0-10.4-7-13.6-12.1-8.8 7-18.1 12.1-32.4 12.1-15.8 0-28.2-9.3-28.2-24.7 0-13.6 9.7-21.4 21.5-24.1 0 .1 37.7-8.9 37.7-9 0-11.6-4.1-16.7-16.3-16.7-10.7 0-16.2 2.9-20.4 9.4-3.4 4.9-2.9 7.8-9.4 7.8-5.1 0-9.6-3.6-9.6-8.8zm32.2 51.9c16.5 0 23.5-8.7 23.5-26.1v-3.7c-4.4 1.5-22.4 6-27.3 6.7-5.2 1-10.4 4.9-10.4 11 .2 6.7 7.1 12.1 14.2 12.1zM354 909c23.3 0 48.6 13.9 48.6 36.1 0 5.7-4.3 10.4-9.9 10.4-7.6 0-8.7-4.1-12.1-9.9-5.6-10.3-12.2-17.2-26.7-17.2-22.3-.2-32.3 19-32.3 42.8 0 24 8.3 41.3 31.4 41.3 15.3 0 23.8-8.9 28.2-20.4 1.8-5.3 4.9-10.4 11.6-10.4 5.2 0 10.4 5.3 10.4 11 0 23.5-24 39.7-48.6 39.7-27 0-42.3-11.4-50.6-30.4-4.1-9.1-6.7-18.4-6.7-31.4-.4-36.4 20.8-61.6 56.7-61.6zm133.3 32.8c6 0 9.4 3.9 9.4 9.9 0 2.4-1.9 7.3-2.7 9.9l-28.7 75.4c-6.4 16.4-11.2 27.7-32.9 27.7-10.3 0-19.3-.9-19.3-9.9 0-5.2 3.9-7.8 9.4-7.8 1 0 2.7.5 3.7.5 1.6 0 2.7.5 3.7.5 10.9 0 12.4-11.2 16.3-18.9l-27.7-68.5c-1.6-3.7-2.7-6.2-2.7-8.4 0-6 4.7-10.4 11-10.4 7 0 9.8 5.5 11.6 11.6l18.3 54.3 18.3-50.2c2.7-7.8 3-15.7 12.3-15.7z" /> </svg>"""

LOGO_SVG = """<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 1400 980" enable-background="new 0 0 1400 980" xml:space="preserve">
<g>
	<g>
		<g>
			<g>
				<g>
					<g>
						<rect fill="#FFFFFF" width="1400" height="980"/>
					</g>
				</g>
			</g>
		</g>
	</g>
	<g>
		<g>
			<g>
				<radialGradient id="SVGID_1_" cx="504.9919" cy="279.0391" r="57.7416" gradientUnits="userSpaceOnUse">
					<stop  offset="0" style="stop-color:#FFFFFF"/>
					<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
					<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
					<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
					<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
					<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
					<stop  offset="1" style="stop-color:#BDBDBD"/>
				</radialGradient>
				<path fill="url(#SVGID_1_)" d="M552.2,292.6c-10,34.6-39.2,56.6-65.3,49c-26.1-7.5-39.1-41.7-29.1-76.3
					c10-34.6,39.2-56.6,65.3-49C549.1,223.9,562.2,258,552.2,292.6z"/>
				<path d="M497.4,345.7c-3.8,0-7.6-0.5-11.2-1.6c-13.4-3.9-24-14.3-29.8-29.5c-5.7-14.9-6.1-32.6-1.1-49.9
					c4.3-14.8,12.4-28.2,22.8-37.7c10.4-9.5,22.6-14.7,34.5-14.7c3.8,0,7.6,0.5,11.2,1.6c27.4,7.9,41.2,43.5,30.9,79.4
					c-4.3,14.8-12.4,28.2-22.8,37.7C521.5,340.5,509.3,345.7,497.4,345.7z M512.6,217.5c-22.2,0-44.2,20.5-52.3,48.7
					c-4.7,16.2-4.3,32.8,1,46.7c5.2,13.6,14.6,23,26.4,26.4c3.2,0.9,6.4,1.4,9.8,1.4c22.2,0,44.2-20.5,52.3-48.7
					c9.6-33.2-2.7-66-27.4-73.1C519.2,217.9,515.9,217.5,512.6,217.5z"/>
			</g>
			<g>
				
					<ellipse transform="matrix(0.8371 0.5471 -0.5471 0.8371 215.7121 -227.4631)" fill="#0B0B0B" cx="489.8" cy="248.4" rx="14.4" ry="16.9"/>
			</g>
			<circle fill="#FFFFFF" cx="484.6" cy="242" r="6.4"/>
			<g>
				<radialGradient id="SVGID_2_" cx="581.7614" cy="283.1665" r="69.1768" gradientUnits="userSpaceOnUse">
					<stop  offset="0" style="stop-color:#FFFFFF"/>
					<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
					<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
					<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
					<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
					<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
					<stop  offset="1" style="stop-color:#BDBDBD"/>
				</radialGradient>
				<path fill="url(#SVGID_2_)" d="M638.3,299.5c-12,41.5-47,67.8-78.2,58.8s-46.8-49.9-34.9-91.4c12-41.5,47-67.8,78.2-58.8
					C634.6,217.1,650.3,258,638.3,299.5z"/>
				<path d="M572.7,362.6c-4.5,0-9-0.6-13.3-1.9c-15.9-4.6-28.4-17-35.3-35.1c-6.8-17.8-7.3-38.9-1.3-59.5
					c5.1-17.7,14.7-33.6,27.1-44.9c12.3-11.3,26.9-17.4,41-17.4c4.5,0,9,0.6,13.3,1.9c15.9,4.6,28.4,17,35.3,35.1
					c6.8,17.8,7.3,38.9,1.3,59.5c-5.1,17.7-14.7,33.6-27.1,44.9C601.3,356.4,586.7,362.6,572.7,362.6z M590.9,208.9
					c-26.8,0-53.4,24.7-63.2,58.7c-5.6,19.5-5.2,39.5,1.2,56.2c6.3,16.5,17.6,27.8,32,32c3.8,1.1,7.8,1.7,11.8,1.7
					c26.8,0,53.4-24.7,63.2-58.7c5.6-19.5,5.2-39.5-1.2-56.2c-6.3-16.5-17.6-27.8-32-32C598.9,209.5,594.9,208.9,590.9,208.9z"/>
			</g>
			<g>
				
					<ellipse transform="matrix(0.8371 0.5471 -0.5471 0.8371 226.6713 -268.127)" fill="#0B0B0B" cx="563.5" cy="246.5" rx="17.3" ry="20.3"/>
			</g>
			<circle fill="#FFFFFF" cx="558.7" cy="238.3" r="6.4"/>
			<g>
				<path fill="#0B0B0B" d="M459.3,192.4c0,0,22.9-35.9,44.1-39.7c18.6-3.3,19.5,12.6,19.5,12.6s15.1-27.1-3.1-35.3
					C501.4,121.8,464.3,152.1,459.3,192.4z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M661.6,198.1c0,0-22.9-35.9-44.1-39.7c-18.6-3.3-19.5,12.6-19.5,12.6s-15.1-27.1,3.1-35.3
					C619.4,127.6,656.6,157.8,661.6,198.1z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M446.8,251.8c0,0,28.1-58.6,69.3-48.7C516.1,203.1,475.7,176.7,446.8,251.8z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M598.7,197.3c0,0,56.1,3.3,53.7,71.8C652.3,269.1,665.5,186.6,598.7,197.3z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M462.9,387.6c0,0,10.2-34.4,33.8-9.1c22.5,24.1,40.4,53.7,67.7,54.5c27.2,0.8,56.1-52,56.1-52
					s-18.2,61.1-56.1,57.8c-38-3.3-58.9-51.6-76.7-61.3C483.2,375.1,478.6,369.4,462.9,387.6z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M601.8,380.1c0,0,24.7-22.2,29.8-5.1c5.1,17.1,5.1,17.1,5.1,17.1S630.8,363.9,601.8,380.1z"/>
			</g>
		</g>
		<g>
			<g>
				<radialGradient id="SVGID_3_" cx="809.6234" cy="272.4345" r="57.7306" gradientUnits="userSpaceOnUse">
					<stop  offset="0" style="stop-color:#FFFFFF"/>
					<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
					<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
					<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
					<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
					<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
					<stop  offset="1" style="stop-color:#BDBDBD"/>
				</radialGradient>
				<path fill="url(#SVGID_3_)" d="M858.6,275.9c-2.6,35.9-26.6,63.5-53.6,61.5c-27.1-1.9-46.9-32.6-44.3-68.6s26.6-63.5,53.6-61.5
					S861.2,240,858.6,275.9z"/>
				<path d="M807.7,340.1c-1,0-1.9,0-2.9-0.1c-28.4-2-49.4-34-46.7-71.3c2.6-35.9,26.1-64,53.5-64c1,0,1.9,0,2.9,0.1
					c28.4,2,49.4,34,46.7,71.3C858.6,312,835.1,340.1,807.7,340.1z M811.6,209.9c-24.8,0-46,26-48.4,59.3c-2.5,34.5,16.4,64,42,65.8
					c0.8,0.1,1.7,0.1,2.5,0.1c24.8,0,46-26,48.4-59.3c2.5-34.5-16.4-64-42-65.8C813.3,209.9,812.4,209.9,811.6,209.9z"/>
			</g>
			<g>
				
					<ellipse transform="matrix(0.9325 0.3611 -0.3611 0.9325 144.6575 -265.6214)" fill="#0B0B0B" cx="783" cy="254.2" rx="14.4" ry="16.9"/>
			</g>
			<circle fill="#FFFFFF" cx="776.4" cy="250.8" r="6.4"/>
			<g>
				<radialGradient id="SVGID_4_" cx="885.574" cy="275.8645" r="69.1636" gradientUnits="userSpaceOnUse">
					<stop  offset="0" style="stop-color:#FFFFFF"/>
					<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
					<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
					<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
					<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
					<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
					<stop  offset="1" style="stop-color:#BDBDBD"/>
				</radialGradient>
				<path fill="url(#SVGID_4_)" d="M944.3,280.1c-3.1,43-31.9,76-64.3,73.7s-56.2-39.1-53.1-82.1c3.1-43,31.9-76,64.3-73.7
					S947.3,237,944.3,280.1z"/>
				<path d="M883.2,356.5c-1.1,0-2.3,0-3.4-0.1c-16.5-1.2-31.4-10.8-41.8-27c-10.3-16-15.2-36.5-13.7-57.9
					c3.1-42.7,31-76.2,63.6-76.2c1.1,0,2.3,0,3.4,0.1c16.5,1.2,31.4,10.8,41.8,27c10.3,16,15.2,36.5,13.7,57.9
					C943.8,323,915.8,356.5,883.2,356.5z M887.9,200.4c-29.9,0-55.6,31.4-58.5,71.5c-1.4,20.3,3.1,39.7,12.8,54.8
					c9.6,14.8,23,23.6,37.9,24.6c1,0.1,2,0.1,3,0.1c29.9,0,55.6-31.4,58.5-71.5c1.4-20.3-3.1-39.7-12.8-54.8
					c-9.6-14.8-23-23.6-37.9-24.6C890,200.4,888.9,200.4,887.9,200.4z"/>
			</g>
			<g>
				
					<ellipse transform="matrix(0.9325 0.3611 -0.3611 0.9325 149.9298 -290.8324)" fill="#0B0B0B" cx="853.1" cy="255.7" rx="17.3" ry="20.3"/>
			</g>
			<circle fill="#FFFFFF" cx="845.6" cy="251.6" r="6.4"/>
			<g>
				<path fill="#0B0B0B" d="M747,257.9c0,0,15.3-63.2,57.7-62.1C804.7,195.8,759.7,178.4,747,257.9z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M832.6,243.8c0,0,18.7-30,51.9-11.8c29.6,16.2,54.6,68.5,54.6,68.5S915,235.8,878.6,224
					S832.6,243.8,832.6,243.8z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M830.4,253.4c0,0-43.9-39.1-64.3-9.1C766.2,244.3,783.3,201.5,830.4,253.4z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M961.3,241.6c0,0-20.2-58.2-55.1-70.6c-34.9-12.4-40.4-6.2-40.4-6.2s-6.2-34.2,18.6-34.2
					C909.3,130.6,949.7,175.6,961.3,241.6z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M857.4,177.1c0,0-36.4-25.4-62-17.8c-25.6,7.6-26.2,13.5-26.2,13.5s-18.1-17.3-3.6-27.8
					C780.1,134.5,822.6,143.6,857.4,177.1z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M956.2,271.6c0,0-2.5,78.4-61.1,88.3C895,360,963.6,366.8,956.2,271.6z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M773.9,391.5c-2.3,3.2,13.6-46.9,58.7-29.6c45.1,17.3,76,56.8,83.4,59.9c0,0-22.9-14.2-37.1-25.3
					S804.2,348.2,773.9,391.5z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M768.8,367.8c0,0-14.6,28,13.1,36.7C781.9,404.4,764.3,396.6,768.8,367.8z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M920.2,404.4c0,0-12,28.4,12.3,32.5C932.6,437,918.7,427.3,920.2,404.4z"/>
			</g>
		</g>
		<g>
			<g>
				<radialGradient id="SVGID_5_" cx="1219.2089" cy="289.2877" r="57.7422" gradientUnits="userSpaceOnUse">
					<stop  offset="0" style="stop-color:#FFFFFF"/>
					<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
					<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
					<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
					<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
					<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
					<stop  offset="1" style="stop-color:#BDBDBD"/>
				</radialGradient>
				<path fill="url(#SVGID_5_)" d="M1172.6,273.9c-11.3,34.2,0.4,68.8,26.2,77.3c25.8,8.5,55.8-12.3,67.1-46.5
					c11.3-34.2-0.4-68.8-26.2-77.3C1213.9,218.9,1183.9,239.7,1172.6,273.9z"/>
				<path d="M1210.7,355.7C1210.7,355.7,1210.7,355.7,1210.7,355.7c-4.3,0-8.6-0.7-12.7-2c-27.1-8.9-39.5-45.1-27.8-80.6
					c9.8-29.6,33.4-50.2,57.6-50.2c4.3,0,8.6,0.7,12.7,2c27.1,8.9,39.5,45.1,27.8,80.6C1258.5,335,1234.8,355.7,1210.7,355.7z
					 M1227.8,228c-22,0-43.7,19.2-52.8,46.7c-10.8,32.8,0.2,66,24.6,74.1c3.6,1.2,7.3,1.8,11.1,1.8c22,0,43.7-19.2,52.8-46.7
					c10.8-32.8-0.2-66-24.6-74.1C1235.3,228.6,1231.6,228,1227.8,228z"/>
			</g>
			<g>
				
					<ellipse transform="matrix(-0.9998 -2.060259e-002 2.060259e-002 -0.9998 2495.3669 590.8484)" fill="#0B0B0B" cx="1250.7" cy="282.6" rx="14.4" ry="16.9"/>
			</g>
			<circle fill="#FFFFFF" cx="1257.4" cy="282.9" r="6.4"/>
			<g>
				<radialGradient id="SVGID_6_" cx="1147.6597" cy="263.5782" r="69.1774" gradientUnits="userSpaceOnUse">
					<stop  offset="0" style="stop-color:#FFFFFF"/>
					<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
					<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
					<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
					<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
					<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
					<stop  offset="1" style="stop-color:#BDBDBD"/>
				</radialGradient>
				<path fill="url(#SVGID_6_)" d="M1091.8,245.1c-13.5,41,0.5,82.4,31.4,92.6C1154,348,1190,323,1203.5,282
					c13.5-41-0.5-82.4-31.4-92.6C1141.3,179.2,1105.3,204.2,1091.8,245.1z"/>
				<path d="M1137.4,342.6C1137.4,342.6,1137.4,342.6,1137.4,342.6c-5.2,0-10.2-0.8-15-2.4c-32.2-10.6-47-53.6-33-95.9
					c11.6-35.2,39.8-59.8,68.5-59.8c5.1,0,10.2,0.8,15,2.4c32.2,10.6,47,53.6,33,95.9C1194.4,318,1166.2,342.6,1137.4,342.6z
					 M1157.9,189.7c-26.6,0-52.8,23.1-63.7,56.3c-13.1,39.6,0.3,79.7,29.8,89.4c4.3,1.4,8.8,2.1,13.4,2.1
					c26.6,0,52.8-23.1,63.7-56.3c13.1-39.6-0.3-79.7-29.8-89.4C1167,190.4,1162.5,189.7,1157.9,189.7z"/>
			</g>
			<g>
				
					<ellipse transform="matrix(-0.9998 -2.060259e-002 2.060259e-002 -0.9998 2365.1091 538.976)" fill="#0B0B0B" cx="1185.3" cy="257.3" rx="17.3" ry="20.3"/>
			</g>
			<circle fill="#FFFFFF" cx="1194.9" cy="262.4" r="6.4"/>
			<g>
				<path fill="#0B0B0B" d="M1282.6,299.6c0,0,9.9-64.2-29.8-79.3C1252.9,220.3,1301.2,221.3,1282.6,299.6z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M1068.5,198c0,0,40.8-46.2,77.8-44.4s39.7,9.6,39.7,9.6s18.7-29.2-4.2-38.7
					C1158.8,115.1,1104.3,141.4,1068.5,198z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M1300.3,241.9c0,0-5.1-43.4-27.5-57.4s-27-10.4-27-10.4s0.9-24.6,18.1-20.9
					C1281.1,157,1302.3,194.4,1300.3,241.9z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M1084,232.8c0,0-27.5,73.5,23,104.9C1107,337.8,1041,318,1084,232.8z"/>
			</g>
			<g>
				<path d="M1246.8,400.4c0,0-0.2,0.4-0.6,1.2c-0.4,0.8-1,2-2,3.5c-0.5,0.7-1,1.5-1.7,2.4c-0.3,0.4-0.7,0.9-1.1,1.3
					c-0.4,0.4-0.8,0.9-1.3,1.3c-1.8,1.8-4,3.8-6.9,5.4c-2.9,1.6-6.3,3.2-10.2,3.9c-0.5,0.1-1,0.2-1.5,0.3c-0.5,0.1-1,0.1-1.5,0.2
					c-0.5,0.1-1,0.1-1.6,0.2c-0.5,0-1.1,0.1-1.6,0.1c-2.2,0-4.4,0.2-6.8-0.3c-4.7-1-9.1-3.1-13.7-5.5c-4.5-2.4-9.1-5.3-13.8-8.3
					c-9.4-6-19.3-12.9-30-19.9c-5.3-3.5-10.8-6.9-16.6-10.2c-5.8-3.3-11.7-6.5-18-9.1c-3.1-1.3-6.3-2.5-9.6-3.4
					c-3.3-0.9-6.6-1.6-9.9-1.8c-3.3-0.2-6.6,0-9.7,0.9c-1.5,0.4-3,1.1-4.4,1.8l-1,0.6l-0.3,0.2l-0.2,0.2l-0.6,0.4
					c-0.8,0.5-1.5,1-2.2,1.5c-5.7,4.2-11,9.1-15.9,14.4c-4.9,5.3-9.5,11-13.6,17c-4.2,6-7.9,12.2-10.9,18.7
					c-1.5,3.2-2.7,6.6-3.5,9.9c-0.4,1.7-0.7,3.3-0.7,4.9c-0.1,1.6,0.1,3,0.6,4.2c0.2,0.5,0.5,1,0.9,1.4c0.4,0.4,0.8,0.7,1.4,0.9
					c1.1,0.5,2.6,0.6,4.1,0.6c1.6-0.1,3.1-0.4,4.8-0.7c1.7-0.4,3.5-0.8,5.2-1.2c7-1.7,13.8-3.4,20.6-5.1c13.5-3.4,26.7-6.3,39.3-8.7
					c12.6-2.4,24.7-4,36-4.4c5.6-0.2,11.1,0,16.2,0.9c2.6,0.4,5.1,1.1,7.4,2c2.3,0.9,4.6,2.1,6.4,3.8c0.5,0.4,0.9,0.8,1.3,1.3
					c0.4,0.4,0.7,0.8,1.1,1.3c0.7,0.9,1.3,1.9,1.9,2.9c0.5,1,0.9,2.1,1.1,3.3c0.1,1.2,0,2.4-0.5,3.4c-0.5,1-1.2,1.9-2,2.5
					c-0.8,0.6-1.6,1.1-2.5,1.5c-1.7,0.8-3.4,1.3-5,1.7c-3.2,0.7-6.1,1-8.7,1.2c-2.5,0.2-4.7,0.2-6.4,0.2c-3.5,0-5.3-0.2-5.3-0.2
					s1.9,0,5.3-0.2c1.7-0.1,3.9-0.3,6.4-0.6c2.5-0.3,5.4-0.8,8.4-1.6c1.5-0.4,3.1-1,4.6-1.8c1.5-0.8,3-1.9,3.6-3.4
					c0.6-1.5,0.2-3.3-0.8-5c-0.5-0.8-1.1-1.7-1.8-2.5c-0.3-0.4-0.7-0.8-1.1-1.2c-0.4-0.3-0.7-0.7-1.1-1c-1.6-1.3-3.6-2.3-5.8-3.1
					c-2.2-0.8-4.5-1.2-7-1.5c-4.9-0.6-10.2-0.6-15.7-0.3c-5.5,0.3-11.2,1-17.1,2c-5.9,0.9-12,2-18.2,3.4
					c-12.4,2.6-25.4,5.9-38.9,9.4c-6.7,1.8-13.6,3.6-20.5,5.4c-1.7,0.4-3.5,0.9-5.2,1.3c-1.8,0.4-3.8,0.8-5.9,1
					c-2.1,0.1-4.3-0.1-6.6-1c-1.1-0.5-2.2-1.2-3.2-2.1c-0.9-0.9-1.6-2-2.1-3.2c-0.9-2.3-1.2-4.5-1.1-6.6c0.1-2.1,0.4-4.1,0.8-6
					c0.9-3.9,2.3-7.5,3.8-11c3.1-7,7-13.5,11.3-19.7c4.3-6.2,9.1-12.1,14.3-17.6c5.2-5.5,10.7-10.6,16.9-15.1
					c0.8-0.6,1.6-1.1,2.3-1.6l0.6-0.4l0.3-0.2l0.3-0.2l1.3-0.8c1.8-0.9,3.7-1.8,5.6-2.3c3.9-1.1,7.9-1.3,11.7-1
					c3.8,0.3,7.5,1.1,11,2.1c3.5,1,6.9,2.3,10.2,3.8c6.5,2.9,12.6,6.2,18.4,9.7c5.8,3.5,11.3,7.1,16.6,10.8
					c10.6,7.2,20.3,14.4,29.4,20.7c4.6,3.2,9,6.1,13.3,8.6c4.3,2.5,8.6,4.7,12.8,5.8c2,0.5,4.2,0.4,6.3,0.5c1,0,2-0.1,3-0.2
					c0.5,0,1-0.1,1.5-0.1c0.5-0.1,1-0.2,1.4-0.2c3.8-0.5,7.1-1.8,10-3.3c2.9-1.4,5.1-3.2,7-4.9c0.5-0.4,0.9-0.8,1.3-1.2
					c0.4-0.4,0.8-0.8,1.1-1.2c0.7-0.8,1.3-1.6,1.9-2.2c1-1.4,1.8-2.5,2.2-3.3C1246.6,400.8,1246.8,400.4,1246.8,400.4z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M1053.1,394.1l28.2-11.8c0,0-16.5,20.7-9.4,19.3c7.1-1.4,37.6-16.5,37.6-14.6s-4.2,14.1-4.2,14.1
					l19.8-3.8c0,0-5.6,6.1-3.8,6.1s24.9-1.4,24.9-1.4l-31,5.2l1.9-5.2c0,0-17.9,4.7-18.3,3.3s1.4-10.3,1.4-10.3s-39,23-38.6,17.4
					s7.1-21.6,4.7-21.2C1063.9,391.7,1053.1,394.1,1053.1,394.1z"/>
			</g>
		</g>
		<g>
			<g>
				<g>
					<g>
						<radialGradient id="SVGID_7_" cx="1105.209" cy="696.1079" r="66.9367" gradientUnits="userSpaceOnUse">
							<stop  offset="0" style="stop-color:#FFFFFF"/>
							<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
							<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
							<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
							<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
							<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
							<stop  offset="1" style="stop-color:#BDBDBD"/>
						</radialGradient>
						<path fill="url(#SVGID_7_)" d="M1048.3,695.8c-0.2,41.8,25.1,75.8,56.5,75.9s57.1-33.5,57.4-75.3
							c0.2-41.8-25.1-75.8-56.5-75.9C1074.2,620.3,1048.5,654,1048.3,695.8z"/>
						<path d="M1105,774.3C1105,774.3,1105,774.3,1105,774.3l-0.3,0c-16-0.1-31-8.4-42.2-23.4c-11-14.8-17-34.4-16.9-55.1
							c0.2-42.9,27-77.8,59.7-77.8l0.3,0c32.8,0.2,59.3,35.4,59.1,78.5C1164.5,739.4,1137.7,774.3,1105,774.3z M1105.4,623.1
							c-29.9,0-54.3,32.6-54.6,72.7c-0.1,19.6,5.5,38.1,15.9,52c10.2,13.7,23.7,21.3,38.1,21.3l0.2,0c29.9,0,54.3-32.6,54.6-72.7
							c0.2-40.3-24-73.2-54-73.4L1105.4,623.1z"/>
					</g>
					<g>
						<circle fill="#0B0B0B" cx="1109.3" cy="702.4" r="20.3"/>
					</g>
					<circle fill="#FFFFFF" cx="1094.3" cy="704.2" r="6.4"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M1053.5,636.9c0,0-41.6,67.4,11.9,126.2C1065.4,763.1,1000.6,709.6,1053.5,636.9z"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M1024.4,687.8c0,0,0,43,28.4,69.4C1052.8,757.2,1015.8,740,1024.4,687.8z"/>
				</g>
			</g>
			<g>
				<g>
					<g>
						<radialGradient id="SVGID_8_" cx="1226.0884" cy="696.1079" r="66.9367" gradientUnits="userSpaceOnUse">
							<stop  offset="0" style="stop-color:#FFFFFF"/>
							<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
							<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
							<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
							<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
							<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
							<stop  offset="1" style="stop-color:#BDBDBD"/>
						</radialGradient>
						<path fill="url(#SVGID_8_)" d="M1283,695.8c0.2,41.8-25.1,75.8-56.5,75.9s-57.1-33.5-57.4-75.3c-0.2-41.8,25.1-75.8,56.5-75.9
							C1257.1,620.3,1282.8,654,1283,695.8z"/>
						<path d="M1226.3,774.3c-32.7,0-59.4-34.9-59.7-77.8c-0.2-43.1,26.3-78.3,59.1-78.5l0.3,0c32.7,0,59.4,34.9,59.7,77.8
							c0.1,20.7-5.9,40.3-16.9,55.1c-11.2,15-26.1,23.3-42.2,23.4L1226.3,774.3z M1225.9,623.1l-0.2,0c-30,0.2-54.2,33.1-54,73.4
							c0.2,40.1,24.7,72.7,54.6,72.7l0.2,0c14.4-0.1,27.9-7.7,38.1-21.3c10.3-13.9,16-32.4,15.9-52
							C1280.3,655.7,1255.8,623.1,1225.9,623.1z"/>
					</g>
					<g>
						<circle fill="#0B0B0B" cx="1222" cy="702.4" r="20.3"/>
					</g>
					<circle fill="#FFFFFF" cx="1207.5" cy="704.2" r="6.4"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M1277.8,636.9c0,0,41.6,67.4-11.9,126.2C1265.9,763.1,1330.7,709.6,1277.8,636.9z"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M1306.9,687.8c0,0,0,43-28.4,69.4C1278.5,757.2,1315.5,740,1306.9,687.8z"/>
				</g>
			</g>
			<g>
				<path fill="#0B0B0B" d="M1305.9,637.6c0,0-16.6-47.9-45.3-58.1c-28.7-10.2-33.2-5.1-33.2-5.1s-5.1-28.1,15.3-28.1
					C1263.1,546.3,1296.3,583.4,1305.9,637.6z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M1021.7,637.6c0,0,16.6-47.9,45.3-58.1c28.7-10.2,33.2-5.1,33.2-5.1s5.1-28.1-15.3-28.1
					S1031.3,583.4,1021.7,637.6z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M1087.1,846.8c1.7-1.7,37.2-46,79.6-40.8c42.4,5.2,49.9,48.3,49.9,48.3s-18.9-27.2-56-28.3
					C1117.5,824.7,1087.1,846.8,1087.1,846.8z"/>
			</g>
		</g>
		<g>
			<g>
				<g>
					<radialGradient id="SVGID_9_" cx="811.7797" cy="675.2575" r="60.7683" gradientUnits="userSpaceOnUse">
						<stop  offset="0" style="stop-color:#FFFFFF"/>
						<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
						<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
						<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
						<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
						<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
						<stop  offset="1" style="stop-color:#BDBDBD"/>
					</radialGradient>
					<path fill="url(#SVGID_9_)" d="M761.2,656.8c-10.9,41.8,10.6,82.9,38.7,85.6c26.9,2.5,53.6-18.6,63.8-57.2
						c7.6-28.7-13.7-27.8-37.2-62.2C803.8,589.9,771.8,616.3,761.2,656.8z"/>
					<path d="M804.3,745.1C804.2,745.1,804.3,745.1,804.3,745.1c-1.5,0-3.1-0.1-4.6-0.2c-11.4-1.1-22.2-8.1-30.3-19.7
						c-12.9-18.4-16.9-44.8-10.6-69.1c7.8-29.9,26.1-50.7,44.4-50.7c6.6,0,16.3,2.8,25.4,16.1c8.5,12.5,16.8,20.2,23.4,26.5
						c11.7,11,19.3,18.2,14.2,37.8C856.6,721.8,832.3,745.1,804.3,745.1z M803.1,610.5c-15.8,0-32.4,19.7-39.4,46.9
						c-6,22.8-2.2,47.7,9.8,64.8c7.3,10.4,16.7,16.6,26.6,17.6c1.4,0.1,2.7,0.2,4.1,0.2c25.7,0,48-21.8,56.9-55.5
						c4.4-16.7-1.3-22.1-12.7-32.8c-6.8-6.4-15.3-14.4-24.1-27.3C818,615.2,810.9,610.5,803.1,610.5z"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M836.5,686.9c-2.4,10.6-11.9,18.2-21.3,17.1c-9.5-1.2-15.3-11.1-12.6-21.9
						c2.6-10.8,12.4-18.2,21.7-16.7C833.5,666.8,838.9,676.4,836.5,686.9z"/>
				</g>
				<circle fill="#FFFFFF" cx="833.1" cy="682.1" r="6.4"/>
			</g>
			<g>
				<g>
					<radialGradient id="SVGID_10_" cx="907.847" cy="685.2764" r="49.8517" gradientUnits="userSpaceOnUse">
						<stop  offset="0" style="stop-color:#FFFFFF"/>
						<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
						<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
						<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
						<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
						<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
						<stop  offset="1" style="stop-color:#BDBDBD"/>
					</radialGradient>
					<path fill="url(#SVGID_10_)" d="M949,680.9c-1.9,35-23.9,59.2-45.7,61.3c-22.7,2.2-38-17.6-36.7-55.2
						c1-28.3,19.7-24.2,47.8-51.1C939.1,612.3,950.8,646.5,949,680.9z"/>
					<path d="M899.9,745c-8.7,0-16.4-3.3-22.4-9.4c-9.6-10-14.2-26.7-13.4-48.6c0.7-19.3,9.3-24.8,22.4-33
						c7.2-4.5,16.1-10.2,26.2-19.8c5.9-5.7,11.6-8.6,16.8-8.6c4.1,0,7.7,1.7,10.9,5c9.7,10.2,12.1,32.7,11.2,50.4
						c-0.9,16.8-6.6,32.6-15.9,44.4c-8.8,11.1-20.5,18.2-32.1,19.3C902.3,744.9,901.1,745,899.9,745z M929.5,630.7
						c-3.9,0-8.3,2.4-13.3,7.1c-10.4,10-19.6,15.8-27,20.4c-12.5,7.9-19.4,12.3-20,28.9c-0.7,20.1,3.5,36,12,44.8
						c5,5.2,11.3,7.8,18.7,7.8c1,0,2.1-0.1,3.2-0.2c19.9-2,41.5-24.5,43.4-58.9c1.1-20.5-2.8-39.2-9.8-46.6
						C934.4,631.8,932.1,630.7,929.5,630.7z"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M890,691.5c-0.7,9.9,5.2,17.5,13.1,17.1c7.8-0.4,14.5-8.5,15.2-18.1c0.6-9.5-5-17.1-12.7-17
						C897.7,673.7,890.7,681.7,890,691.5z"/>
				</g>
				<circle fill="#FFFFFF" cx="915.2" cy="687.5" r="6.4"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M948.5,584.9c-18.3,11.6-59.4,65.9-76.2,78.6c-0.5,0.2-1.1,0.3-1.6,0.4c-10.4,2.5-46.8-85.3-69.1-108
					c-22.9-23.4-60,6.8-60,6.8s5.9,2.5,29.9,18.3c23.3,15.4,64.9,82.6,81.9,95.4c2,1.5,4,2.6,5.9,3.2c4.2,3.2,10.4,4.9,18.8,0.4
					c17.9-9.6,65.8-60.5,84.8-68.3c18.2-7.5,22.2-8.2,22.2-8.2S968.8,571.9,948.5,584.9z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M746.9,670.8c0,0-17.7,83.9,58.5,77.9C805.4,748.8,746.3,751.7,746.9,670.8z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M896.3,748.2c0,0,46.1,3,59.6-50.2C956,698,955.4,757.6,896.3,748.2z"/>
			</g>
			<g>
				<path d="M938.6,761.2c0,0-0.1,0.5-0.4,1.3c-0.3,0.9-0.8,2.1-1.7,3.6c-1.7,3-5,7.3-10.7,10.9c-2.8,1.8-6.1,3.4-9.9,4.8
					c-3.8,1.3-8,2.3-12.6,3c-9.2,1.3-19.7,1.2-30.9-0.2l-1.1-0.1l-0.6-0.1l-0.5-0.1c-0.7-0.1-1.4-0.2-2.1-0.3
					c-1.4-0.2-2.8-0.4-4.3-0.6c-2.9-0.4-5.8-0.9-8.8-1.4c-6-1-12.1-2.3-18.3-3.7c-6.3-1.4-12.7-2.9-19.2-4.5
					c-6.5-1.6-13.1-3.3-19.8-4.6c-3.3-0.7-6.7-1.3-10-1.7c-3.3-0.4-6.6-0.6-9.7-0.3c-3.1,0.3-5.9,1-8.3,2.4
					c-0.6,0.4-1.2,0.7-1.7,1.2l-0.8,0.6l-0.8,0.7l-0.4,0.4c-0.1,0.1,0,0-0.1,0.1l-0.1,0.1l-0.2,0.3c-0.3,0.4-0.6,0.6-0.8,0.9
					c-0.5,0.6-1.1,1.3-1.6,1.9c-4.3,5.4-8.2,11.5-12,17.6c-3.7,6.2-7.1,12.6-10.1,19.2c-1.5,3.3-2.9,6.6-4.1,9.9
					c-1.2,3.3-2.3,6.7-3,9.9c-0.4,1.6-0.6,3.3-0.8,4.8c-0.1,1.5-0.1,3,0.1,4.1c0.1,0.6,0.3,1,0.4,1.4c0.2,0.3,0.3,0.5,0.5,0.7
					c0.1,0.2,0.3,0.3,0.7,0.4c0.1,0.1,0.4,0.1,0.6,0.2l0.4,0l0.1,0c0,0,0.1,0,0,0l0.3,0c0.6,0,1.4,0,2.2,0c0.8,0,1.6-0.1,2.5-0.1
					c1.7-0.1,3.4-0.3,5.2-0.6c3.5-0.4,7-1,10.5-1.6c7-1.2,13.9-2.6,20.7-3.9c13.6-2.7,26.8-5.3,39.3-7.8c6.3-1.2,12.4-2.4,18.3-3.5
					c5.9-1.1,11.6-2.1,17.2-2.9c11-1.8,21.2-3.2,30.4-3.7c2.3-0.1,4.5-0.2,6.7-0.1c2.2,0,4.3,0.2,6.4,0.6c1,0.2,2.1,0.5,3.1,1.2
					c0.5,0.3,1,0.8,1.4,1.4c0.1,0.2,0.2,0.3,0.2,0.5l0.1,0.3l0.2,0.6c0.7,1.9,0.8,3.8,0.5,5.6c-0.3,1.8-1,3.3-1.8,4.6
					c-0.8,1.3-1.8,2.3-2.7,3.2c-1.9,1.8-3.7,3-5.2,3.9c-1.5,0.9-2.7,1.4-3.6,1.8c-0.8,0.4-1.3,0.5-1.3,0.5s0.4-0.2,1.2-0.6
					c0.8-0.4,1.9-1.1,3.3-2.2c1.4-1,3.1-2.4,4.7-4.2c0.8-0.9,1.6-2,2.2-3.2c0.6-1.2,1.1-2.6,1.2-4c0.1-1.4-0.1-3-0.8-4.4l-0.3-0.6
					l-0.1-0.3c0,0-0.1-0.1-0.1-0.1c-0.1-0.1-0.3-0.3-0.6-0.5c-0.6-0.3-1.4-0.5-2.3-0.6c-3.6-0.4-7.8-0.1-12.2,0.5
					c-4.4,0.6-9.1,1.4-14.1,2.3c-4.9,1-10.1,2-15.6,3.2c-5.4,1.2-11,2.5-16.9,3.9c-5.8,1.4-11.9,2.8-18,4.3
					c-12.4,3-25.4,6.2-39.1,9.2c-6.8,1.5-13.7,3-20.9,4.4c-3.6,0.7-7.2,1.3-10.8,1.9c-1.8,0.3-3.7,0.5-5.6,0.7c-1,0.1-1.9,0.2-3,0.2
					c-1,0-2,0.1-3.3,0c-2.3-0.1-5.6-0.9-8-3.1c-1.2-1.1-2.1-2.4-2.8-3.7c-0.6-1.3-1-2.6-1.3-3.9c-0.5-2.5-0.5-4.7-0.3-6.9
					c0.2-2.1,0.5-4.2,0.9-6.1c0.9-3.9,2-7.7,3.4-11.3c1.3-3.6,2.8-7.2,4.4-10.6c3.2-6.9,6.8-13.6,10.7-20.1
					c3.9-6.5,8.1-12.7,13-18.7c0.6-0.7,1.2-1.5,1.9-2.2c0.3-0.4,0.7-0.8,1-1.1l0.2-0.2l0.1-0.1c0,0,0.2-0.2,0.2-0.2l0.6-0.6l1.2-1.1
					c0.4-0.4,0.9-0.7,1.3-1c0.9-0.7,1.9-1.3,2.8-1.8c4-2.2,8.3-3.1,12.4-3.4c4.1-0.3,7.9,0.1,11.6,0.6c3.7,0.5,7.2,1.3,10.7,2.1
					c6.9,1.6,13.5,3.5,20,5.4c6.4,1.8,12.7,3.6,18.8,5.2c6.1,1.6,12.1,3.1,17.9,4.4c2.9,0.7,5.8,1.3,8.5,1.9
					c1.4,0.3,2.8,0.6,4.2,0.9c0.7,0.1,1.4,0.3,2.1,0.4l0.5,0.1l0.5,0.1l1,0.2c5.4,1,10.6,1.6,15.5,2c5,0.4,9.7,0.5,14.1,0.1
					c8.8-0.6,16.4-2.6,22-5.6c5.6-3,9.2-6.6,11.2-9.4c1-1.4,1.6-2.6,2-3.4C938.5,761.6,938.6,761.2,938.6,761.2z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M895.1,784.9l-8.2,14.1l-15.1,1.4l-6.9,5.2l-0.7-5.8c0,0-28.5,8.6-26.5,8.6c2.1,0,24.4-5.5,24.4-5.5
					s-1.4,11.3-0.7,10.3s10.6-11,12-11s8.9,0.7,8.9,0.7l-3.4,18.9l5.5-20.3l4.8-0.7L895.1,784.9z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M761.7,778.9c0,0-0.3,23.6,1.2,23c1.5-0.6,23.9-12.7,23.9-12.7s-1.5,12.4,0.9,12.4s27.2-7.4,27.2-7.4
					l-5.3,9.4l9.4,0.9l-13.3,2.1l2.7-5.6l-21.6,4.7l-2.4-9.2L762,812.3c0,0-20.1,17.4-18.9,16.2s15.9-17.4,15.9-17.4s-7.1,3-6.2,1.5
					C753.7,811.1,761.7,778.9,761.7,778.9z"/>
			</g>
		</g>
		<g>
			<g>
				<g>
					<radialGradient id="SVGID_11_" cx="494.6841" cy="718.4818" r="56.4264" gradientUnits="userSpaceOnUse">
						<stop  offset="0" style="stop-color:#FFFFFF"/>
						<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
						<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
						<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
						<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
						<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
						<stop  offset="1" style="stop-color:#BDBDBD"/>
					</radialGradient>
					<path fill="url(#SVGID_11_)" d="M447.1,661.7c-5.3,11.3-8.4,24.6-8.5,38.8c-0.2,41.1,24.7,74.6,55.6,74.7
						c31,0.2,56.2-33,56.5-74.1c0-6.1-0.5-12.1-1.5-17.8C513.7,671.3,472,664.7,447.1,661.7z"/>
					<path d="M494.5,777.8C494.5,777.8,494.5,777.8,494.5,777.8l-0.3,0c-32.3-0.2-58.4-34.9-58.2-77.3c0.1-14.1,3.1-27.9,8.7-39.9
						l0.8-1.7l1.8,0.2c22.4,2.7,65.6,9.2,102.6,21.8l1.4,0.5l0.3,1.5c1.1,6,1.6,12.1,1.6,18.3C553.1,743.4,526.7,777.8,494.5,777.8z
						 M448.7,664.5c-4.8,10.9-7.4,23.4-7.5,36.1c-0.2,39.6,23.6,72,53.1,72.2l0.2,0c29.4,0,53.5-32.1,53.7-71.6
						c0-5.3-0.4-10.7-1.2-15.9C511.7,673.5,471,667.2,448.7,664.5z"/>
				</g>
				<g>
					<circle fill="#0B0B0B" cx="498.7" cy="695.6" r="20"/>
				</g>
				<circle fill="#FFFFFF" cx="509.4" cy="686" r="8.4"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M553.1,668.2c0,0-6.1,9.1-40.9-1.5c-34.8-10.6-117.4-31.8-117.4-31.8c2.3-4.5,34.1-28,49.2-23.5
					C459.2,616,527.3,669.7,553.1,668.2z"/>
			</g>
			<g>
				<g>
					<radialGradient id="SVGID_12_" cx="613.7263" cy="718.4818" r="56.4264" gradientUnits="userSpaceOnUse">
						<stop  offset="0" style="stop-color:#FFFFFF"/>
						<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
						<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
						<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
						<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
						<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
						<stop  offset="1" style="stop-color:#BDBDBD"/>
					</radialGradient>
					<path fill="url(#SVGID_12_)" d="M661.3,661.7c5.3,11.3,8.4,24.6,8.5,38.8c0.2,41.1-24.7,74.6-55.6,74.7
						c-31,0.2-56.2-33-56.5-74.1c0-6.1,0.5-12.1,1.5-17.8C594.8,671.3,636.4,664.7,661.3,661.7z"/>
					<path d="M613.9,777.8c-32.2,0-58.6-34.4-58.8-76.7c0-6.2,0.5-12.3,1.6-18.3l0.3-1.5l1.4-0.5c37-12.6,80.2-19.1,102.6-21.8
						l1.8-0.2l0.8,1.7c5.6,12,8.6,25.8,8.7,39.9c0.1,20.4-5.8,39.7-16.6,54.3c-11,14.8-25.8,23-41.6,23.1L613.9,777.8z M561.5,685.3
						c-0.8,5.2-1.3,10.5-1.2,15.9c0.2,39.5,24.3,71.6,53.7,71.6l0.2,0c14.2-0.1,27.5-7.5,37.5-21c10.2-13.7,15.7-31.8,15.6-51.2
						c-0.1-12.7-2.6-25.2-7.5-36.1C637.4,667.2,596.7,673.5,561.5,685.3z"/>
				</g>
				<g>
					<circle fill="#0B0B0B" cx="609.7" cy="695.6" r="20"/>
				</g>
				<circle fill="#FFFFFF" cx="620.2" cy="686" r="8.4"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M555.3,668.2c0,0,6.1,9.1,40.9-1.5c34.8-10.6,117.4-31.8,117.4-31.8c-2.3-4.5-34.1-28-49.2-23.5
					C649.2,616,581.1,669.7,555.3,668.2z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M436.4,742.5c0,0,13.8,79.2,87,32.7C523.3,775.2,461.3,808.8,436.4,742.5z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M668.8,742.5c0,0-13.8,79.2-87,32.7C581.9,775.2,643.9,808.8,668.8,742.5z"/>
			</g>
		</g>
		<g>
			<g>
				<g>
					<g>
						<radialGradient id="SVGID_13_" cx="186.8046" cy="275.4618" r="66.9941" gradientUnits="userSpaceOnUse">
							<stop  offset="0" style="stop-color:#FFFFFF"/>
							<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
							<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
							<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
							<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
							<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
							<stop  offset="1" style="stop-color:#BDBDBD"/>
						</radialGradient>
						<path fill="url(#SVGID_13_)" d="M129.8,275.1c-0.2,41.8,25.1,75.8,56.6,76c31.5,0.2,57.2-33.6,57.4-75.4
							c0.2-41.8-25.1-75.8-56.6-76C155.8,199.6,130,233.3,129.8,275.1z"/>
						<path d="M186.6,353.7C186.6,353.7,186.6,353.7,186.6,353.7l-0.3,0c-16-0.1-31-8.4-42.2-23.4c-11-14.8-17-34.4-16.9-55.1
							c0.2-43,27-77.9,59.7-77.9l0.3,0c16,0.1,31,8.4,42.2,23.4c11,14.8,17,34.4,16.9,55.1C246.1,318.8,219.3,353.7,186.6,353.7z
							 M187,202.3c-29.9,0-54.4,32.7-54.6,72.8c-0.1,19.7,5.5,38.2,15.9,52.1c10.2,13.7,23.7,21.3,38.1,21.4l0.2,0
							c29.9,0,54.4-32.7,54.6-72.8c0.1-19.7-5.5-38.2-15.9-52.1c-10.2-13.7-23.7-21.3-38.1-21.4L187,202.3z"/>
					</g>
					<g>
						<circle fill="#0B0B0B" cx="190.9" cy="234.5" r="20.3"/>
					</g>
					<circle fill="#FFFFFF" cx="198.2" cy="228.4" r="7.7"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M135,216.2c0,0-41.7,67.5,11.9,126.3C146.9,342.5,82.1,288.9,135,216.2z"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M105.9,267.1c0,0,0,43,28.4,69.4C134.4,336.6,97.3,319.4,105.9,267.1z"/>
				</g>
			</g>
			<g>
				<g>
					<g>
						<radialGradient id="SVGID_14_" cx="299.3604" cy="288.2467" r="60.747" gradientUnits="userSpaceOnUse">
							<stop  offset="0" style="stop-color:#FFFFFF"/>
							<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
							<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
							<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
							<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
							<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
							<stop  offset="1" style="stop-color:#BDBDBD"/>
						</radialGradient>
						<path fill="url(#SVGID_14_)" d="M351,288c0.2,37.9-22.8,68.8-51.3,68.9c-28.5,0.2-51.9-30.4-52.1-68.3s22.8-68.8,51.3-68.9
							C327.5,219.5,350.8,250.1,351,288z"/>
						<path d="M299.5,359.4c-29.8,0-54.2-31.8-54.4-70.9c-0.2-39.2,23.9-71.3,53.8-71.5l0.2,0c29.8,0,54.2,31.8,54.4,70.9
							c0.1,18.9-5.4,36.7-15.4,50.2c-10.2,13.7-23.8,21.3-38.5,21.3L299.5,359.4z M299.2,222.2l-0.2,0c-27.1,0.2-49,29.9-48.8,66.3
							c0.2,36.3,22.3,65.8,49.3,65.8l0.2,0c13-0.1,25.2-6.9,34.4-19.3c9.4-12.6,14.5-29.3,14.4-47.1
							C348.3,251.7,326.2,222.2,299.2,222.2z"/>
					</g>
					<g>
						<circle fill="#0B0B0B" cx="295.6" cy="251.1" r="18.4"/>
					</g>
					<circle fill="#FFFFFF" cx="302.3" cy="248.8" r="7.7"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M355.1,256.5c0,0,21.5,68.6-38.7,108.4C316.4,364.9,385.4,332.3,355.1,256.5z"/>
				</g>
				<g>
					<path fill="#0B0B0B" d="M369.3,307.8c0,0-9.6,37.8-40.5,54.7C328.8,362.5,365.2,355.6,369.3,307.8z"/>
				</g>
			</g>
			<g>
				<path fill="#0B0B0B" d="M370.3,265.1c0,0-15.1-43.5-41.1-52.7c-26.1-9.3-30.1-4.6-30.1-4.6s-4.6-25.5,13.9-25.5
					S361.6,215.9,370.3,265.1z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M126.4,216.9c0,0,16.6-47.9,45.4-58.2s33.2-5.1,33.2-5.1s5.1-28.1-15.3-28.1S135.9,162.6,126.4,216.9z"
					/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M320.7,409.7c0-3.5-8.8,29.2-66.1,11.6c-65.5-20.2-135.5-104.1-125.6-52.9
					C138.9,419.7,321.6,480.8,320.7,409.7z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M121,348.5c0,0-30.4,43.4,63.9,77.5C184.8,426.1,109.8,390.7,121,348.5z"/>
			</g>
		</g>
		<g>
			<g>
				<g>
					<g>
						<radialGradient id="SVGID_15_" cx="176.664" cy="694.6469" r="69.1721" gradientUnits="userSpaceOnUse">
							<stop  offset="0" style="stop-color:#FFFFFF"/>
							<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
							<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
							<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
							<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
							<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
							<stop  offset="1" style="stop-color:#BDBDBD"/>
						</radialGradient>
						<path fill="url(#SVGID_15_)" d="M118.8,683.9c-7.9,42.4,11.6,81.6,43.6,87.6s64.2-23.6,72.1-66.1
							c7.9-42.4-11.6-81.6-43.6-87.6S126.7,641.5,118.8,683.9z"/>
						<path d="M170.6,774.8C170.6,774.8,170.6,774.8,170.6,774.8c-2.9,0-5.8-0.3-8.7-0.8c-16.3-3-30-14.2-38.5-31.5
							c-8.5-17-11-38-7.1-59.1c3.6-19.1,12.2-36.6,24.4-49.3c12.2-12.7,27.1-19.6,42-19.6c2.9,0,5.8,0.3,8.7,0.8
							c16.3,3,30,14.2,38.5,31.5c8.5,17,11,38,7.1,59.1c-3.6,19.1-12.2,36.6-24.4,49.3C200.4,767.8,185.5,774.8,170.6,774.8z
							 M182.7,619.6c-28.1,0-54.5,27.8-61.4,64.7c-3.7,20-1.4,39.8,6.6,55.9c7.9,15.8,20.3,26,34.9,28.7c2.6,0.5,5.2,0.7,7.8,0.7
							c28.1,0,54.5-27.8,61.4-64.7c3.7-20,1.4-39.8-6.6-55.9c-7.9-15.8-20.3-26-34.9-28.7C187.9,619.9,185.3,619.6,182.7,619.6z"/>
					</g>
					<g>
						<circle fill="#0B0B0B" cx="179.7" cy="701.8" r="21"/>
					</g>
					<circle fill="#FFFFFF" cx="186.2" cy="709.9" r="8.4"/>
				</g>
			</g>
			<g>
				<g>
					<g>
						<radialGradient id="SVGID_16_" cx="301.5597" cy="694.6469" r="69.1735" gradientUnits="userSpaceOnUse">
							<stop  offset="0" style="stop-color:#FFFFFF"/>
							<stop  offset="0.4408" style="stop-color:#FDFDFD"/>
							<stop  offset="0.6288" style="stop-color:#F6F6F6"/>
							<stop  offset="0.7688" style="stop-color:#E9E9E9"/>
							<stop  offset="0.8844" style="stop-color:#D8D8D8"/>
							<stop  offset="0.9847" style="stop-color:#C1C1C1"/>
							<stop  offset="1" style="stop-color:#BDBDBD"/>
						</radialGradient>
						<path fill="url(#SVGID_16_)" d="M359.2,682.6c8.8,42.2-9.8,81.9-41.6,88.5c-31.8,6.7-64.8-22.2-73.6-64.4s9.8-81.9,41.6-88.5
							S350.3,640.3,359.2,682.6z"/>
						<path d="M308.3,774.6c-14.7,0-29.5-6.8-41.8-19.1c-12.2-12.3-21.2-29.5-25.1-48.3c-4.4-21-2.4-42,5.7-59.2
							c8.2-17.5,21.6-29,37.8-32.3c3.2-0.7,6.5-1,9.8-1c14.7,0,29.5,6.8,41.8,19.1c12.2,12.3,21.2,29.5,25.1,48.3
							c9.1,43.5-10.4,84.6-43.6,91.6C314.9,774.3,311.6,774.6,308.3,774.6z M294.8,619.8c-2.9,0-5.9,0.3-8.7,0.9
							c-14.6,3.1-26.7,13.5-34.2,29.5c-7.6,16.2-9.5,36.1-5.4,56c3.7,17.8,12.2,34.1,23.7,45.7c11.3,11.4,24.8,17.6,38.1,17.6
							c2.9,0,5.9-0.3,8.7-0.9c30.4-6.4,48.1-44.7,39.6-85.5c-3.7-17.8-12.2-34.1-23.7-45.7C321.6,626,308.1,619.8,294.8,619.8z"/>
					</g>
					<g>
						<circle fill="#0B0B0B" cx="298.7" cy="701.8" r="21"/>
					</g>
					<circle fill="#FFFFFF" cx="305.1" cy="709.9" r="8.4"/>
				</g>
			</g>
			<g>
				<path fill="#0B0B0B" d="M348.2,637.6c0,0-17.2-49.5-46.8-60c-29.7-10.6-34.3-5.3-34.3-5.3s-5.3-29,15.8-29
					C304,543.3,338.3,581.6,348.2,637.6z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M119.4,642.8c0,0,17.2-49.5,46.8-60c29.7-10.6,34.3-5.3,34.3-5.3s5.3-29-15.8-29
					S129.3,586.7,119.4,642.8z"/>
			</g>
			<g>
				<polygon fill="#0B0B0B" points="91.8,683.9 234,683.9 233.2,675.9 				"/>
			</g>
			<g>
				<polygon fill="#0B0B0B" points="243.3,682.2 383.1,682.2 244.2,675 				"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M128.9,857.4c0,0,101.4-57,188.1-27.8l-1.3-19.7C315.6,809.9,225.6,789.4,128.9,857.4z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M317,792.7c0,0-19.5,42.4,24.4,54.7C341.4,847.4,314.8,830,317,792.7z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M106.2,736.5c0,0,9,68.8,77.8,42.6C184,779.1,124.9,797.1,106.2,736.5z"/>
			</g>
			<g>
				<path fill="#0B0B0B" d="M295.5,782.8c0,0,49.4,17.2,71.1-46.4C366.5,736.5,345.6,785.8,295.5,782.8z"/>
			</g>
		</g>
	</g>
</g>
</svg>"""


LOGO = get_svg(LOGO_SVG, wrap=False, style="max-width: 100%; margin-bottom: 25px")
