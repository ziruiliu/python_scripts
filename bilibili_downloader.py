#!/usr/bin/env python3
"""
Bilibili Video Downloader

This module provides functionality to download videos from bilibili.com.
It uses the bilibili API to extract video information and download the content.

Usage:
    python bilibili_downloader.py <video_url> [output_path]

Examples:
    python bilibili_downloader.py https://www.bilibili.com/video/BV1xN6xBjEqy
    python bilibili_downloader.py https://www.bilibili.com/video/BV1xN6xBjEqy -o video.mp4
    python bilibili_downloader.py https://www.bilibili.com/video/BV1xN6xBjEqy -q 80
"""

import re
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import requests
from urllib.parse import urlparse, parse_qs
import time
import hashlib
import os


class BilibiliDownloader:
    """Downloader for Bilibili videos."""

    # Bilibili API endpoints
    API_PLAY_URL = "https://api.bilibili.com/x/player/wbi/playurl"
    API_VIDEO_INFO = "https://api.bilibili.com/x/web-interface/view"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.bilibili.com',
        })
        self.buvid = self._generate_buvid()

    def _generate_buvid(self) -> str:
        """Generate a pseudo-unique buvid (browser unique identifier)."""
        import uuid
        return f"XY{uuid.uuid4().hex.upper()}"

    def _get_cid(self, url: str) -> Optional[str]:
        """
        Extract CID (content ID) from bilibili video URL.

        Args:
            url: The URL of the bilibili video page

        Returns:
            The CID string or None
        """
        try:
            # Extract BV number from URL
            match = re.search(r'([bB][vV][a-zA-Z0-9]+)', url)
            if not match:
                print(f"Could not find BV number in URL: {url}")
                return None

            bv_id = match.group(1)

            # Use the web-interface/view API to get CID
            params = {
                'bvid': bv_id,
            }

            response = self.session.get(self.API_VIDEO_INFO, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == 0:
                return str(data['data']['cid'])
            else:
                print(f"API returned error: {data.get('message', 'Unknown error')}")
                return None

        except Exception as e:
            print(f"Error getting CID: {e}")
            return None

    def _get_play_url(self, cid: str, bv_id: str, quality: int = 80) -> Optional[Dict[str, Any]]:
        """
        Get the play URL for a video using the Bilibili API.

        Args:
            cid: Content ID
            bv_id: BV identifier
            quality: Desired quality code

        Returns:
            Play URL data or None
        """
        try:
            # Build query parameters
            params = {
                'bvid': bv_id,
                'cid': cid,
                'qn': quality,
                'fnver': 0,
                'fnval': 4048,  # 4k + dash + no watermark
                'otype': 'json',
            }

            response = self.session.get(self.API_PLAY_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == 0:
                return data['data']
            else:
                print(f"Play URL API returned error: {data.get('message', 'Unknown error')}")
                return None

        except Exception as e:
            print(f"Error getting play URL: {e}")
            return None

    def extract_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract video information from a bilibili video page.

        Args:
            url: The URL of the bilibili video page

        Returns:
            Dictionary containing video information or None if extraction fails
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html_content = response.text

            # Look for the video data embedded in the page
            # Bilibili often embeds video info in window.__INITIAL_STATE__ or similar
            patterns = [
                # Pattern for video data in JSON format
                r'window\.__INITIAL_STATE__\s*=\s*({.+?});\s*window\.',
                r'window\.__playinfo__\s*=\s*({.+?});\s*window\.',
                r'"dash":\s*({[^}]+(?:}[^}]+){1,10})',  # Dash format data
            ]

            video_info = None

            for pattern in patterns:
                match = re.search(pattern, html_content)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        if 'videoData' in data or 'dash' in str(data):
                            video_info = data
                            break
                    except json.JSONDecodeError:
                        continue

            # If not found in initial state, try to extract dash data directly
            if not video_info:
                dash_match = re.search(r'"dash":\s*({[^}]+(?:}[^}]+){1,15})', html_content)
                if dash_match:
                    # Try to extract video data from the full page
                    video_data_match = re.search(r'"videoData":\s*({[^}]+(?:}[^}]+){1,5})', html_content)
                    if video_data_match:
                        try:
                            video_data = json.loads(video_data_match.group(1))
                            # Find dash data in the full response
                            dash_data_match = re.search(r'"dash":\s*({[^}]+})', html_content)
                            if dash_data_match:
                                dash_data = json.loads(dash_data_match.group(1))
                                video_info = {
                                    'videoData': video_data,
                                    'dash': dash_data
                                }
                        except json.JSONDecodeError:
                            pass

            return video_info

        except requests.RequestException as e:
            print(f"Error fetching video page: {e}")
            return None

    def get_video_streams(self, video_info: Dict[str, Any]) -> list:
        """
        Extract available video streams from video info.

        Args:
            video_info: The video information dictionary

        Returns:
            List of available video streams with their quality and URLs
        """
        streams = []

        if not video_info:
            return streams

        # Try to get dash data (modern format)
        dash = video_info.get('dash', {}) or video_info.get('data', {}).get('dash', {})

        if dash:
            # Get video streams
            video_streams = dash.get('video', [])
            for stream in video_streams:
                streams.append({
                    'type': 'video',
                    'quality': stream.get('quality', 'unknown'),
                    'format': stream.get('format', 'mp4'),
                    'duration': stream.get('duration', 0),
                    'url': stream.get('baseUrl', stream.get('base_url', '')),
                    'bandwidth': stream.get('bandwidth', 0),
                    'codecs': stream.get('codecs', []),
                })

        # Try to getflv streams (legacy format)
        durl = video_info.get('durl', []) or video_info.get('data', {}).get('durl', [])
        for stream in durl:
            streams.append({
                'type': 'video',
                'format': 'flv',
                'url': stream.get('url', ''),
                'length': stream.get('length', 0),
            })

        return streams

    def download_video(self, url: str, output_path: Optional[str] = None,
                      quality: Optional[int] = None) -> Optional[str]:
        """
        Download a bilibili video.

        Args:
            url: The URL of the bilibili video page
            output_path: Path to save the video (optional)
            quality: Desired video quality (optional)

        Returns:
            Path to downloaded video or None if download fails
        """
        print(f"Fetching video info from: {url}")

        # Extract BV ID from URL
        match = re.search(r'([bB][vV][a-zA-Z0-9]+)', url)
        if not match:
            print(f"Could not find BV number in URL: {url}")
            return None
        bv_id = match.group(1)

        # Get CID using the web-interface/view API
        cid = self._get_cid(url)
        if not cid:
            print("Failed to get video CID")
            return None

        print(f"Video CID: {cid}")

        # Use default quality if not specified
        if not quality:
            quality = 80  # Default to 1080P

        # Get play URL using the playurl API
        play_data = self._get_play_url(cid, bv_id, quality)
        if not play_data:
            print("Failed to get play URL")
            return None

        # Print video information
        video_data = play_data.get('videoData', {})
        if not video_data:
            # Try to get from dash data
            video_info_resp = self.session.get(self.API_VIDEO_INFO, params={'bvid': bv_id}, timeout=10)
            if video_info_resp.status_code == 200:
                info_data = video_info_resp.json()
                if info_data.get('code') == 0:
                    video_data = info_data['data']

        print(f"Video title: {video_data.get('title', 'Unknown')}")
        print(f"Author: {video_data.get('owner', {}).get('name', 'Unknown')}")

        # Get streams from dash data
        dash = play_data.get('dash', {})
        video_streams = dash.get('video', [])
        audio_streams = dash.get('audio', [])

        if not video_streams:
            print("No video streams found")
            return None

        # Sort by bandwidth and select best
        video_streams.sort(key=lambda x: x.get('bandwidth', 0), reverse=True)
        best_stream = video_streams[0]

        print(f"Best quality available: {best_stream.get('quality', 'unknown')} "
              f"({best_stream.get('codecs', ['unknown'])[0] if best_stream.get('codecs') else 'unknown'})")

        # Filter by quality if specified
        if quality:
            filtered = [s for s in video_streams if s.get('quality') == quality]
            if filtered:
                best_stream = filtered[0]

        # Generate output filename
        if not output_path:
            title = video_data.get('title', 'video').replace('/', '-').replace('\\', '-')
            output_path = f"{title}.mp4"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Download video stream
        video_url = best_stream.get('baseUrl', best_stream.get('base_url', ''))
        if not video_url:
            print("No video URL found")
            return None

        print(f"Downloading video to: {output_file}")
        video_file = self._download_file(video_url, str(output_file) + ".video.mp4")

        # Download audio if available (DASH format)
        audio_file = None
        if audio_streams:
            audio_streams.sort(key=lambda x: x.get('bandwidth', 0), reverse=True)
            best_audio = audio_streams[0]
            audio_url = best_audio.get('baseUrl', best_audio.get('base_url', ''))
            if audio_url:
                print(f"Downloading audio...")
                audio_file = self._download_file(audio_url, str(output_file) + ".audio.m4a")

        # Merge video and audio if audio was downloaded
        if audio_file and os.path.exists(audio_file):
            print("Merging video and audio...")
            merged_file = str(output_file) + ".merged.mp4"
            try:
                # Use ffmpeg to merge video and audio
                result = subprocess.run([
                    'ffmpeg', '-i', video_file, '-i', audio_file,
                    '-c', 'copy', merged_file,
                    '-y'
                ], capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(merged_file):
                    # Replace original with merged file
                    os.remove(video_file)
                    os.remove(audio_file)
                    os.rename(merged_file, output_file)
                    print(f"Download complete: {output_file}")
                    return str(output_file)
                else:
                    print(f"Merge failed, keeping separate files")
            except FileNotFoundError:
                print("ffmpeg not found, keeping video and audio separate")
            except Exception as e:
                print(f"Merge error: {e}")

        # If no audio or merge failed, just rename video file
        if video_file and os.path.exists(video_file):
            os.rename(video_file, output_file)
            print(f"Download complete: {output_file}")
            return str(output_file)

        return None

    def _download_file(self, url: str, output_path: str, chunk_size: int = 8192) -> Optional[str]:
        """
        Download a file from the given URL.

        Args:
            url: The URL to download from
            output_path: Path to save the file
            chunk_size: Size of download chunks

        Returns:
            Path to downloaded file or None if download fails
        """
        try:
            print(f"Downloading from: {url}")
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            print(f"Download complete: {output_path} ({downloaded_size // 1024} KB)")
            return output_path

        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return None


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Download videos from bilibili.com'
    )
    parser.add_argument('url', help='URL of the bilibili video')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-q', '--quality', type=int,
                        help='Download quality (e.g., 80 for 1080P)')

    args = parser.parse_args()

    downloader = BilibiliDownloader()
    result = downloader.download_video(args.url, args.output, args.quality)

    if result:
        print(f"\nSuccess! Video saved to: {result}")
    else:
        print("\nDownload failed!")
        exit(1)


if __name__ == '__main__':
    main()
